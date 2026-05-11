import re
from typing import List, Dict, Any
from dataclasses import dataclass, field
from typing import Dict, List, Any
import uuid
import networkx as nx

class STGAdapter:
    def __init__(self, instructions: str = "", role: str = "Formula Student Race Engineer"):
        self.role = role
        self.instructions = instructions
        self.FORMAT_MAP = {
            'rpm_mean':     ('RPM', '{:.0f}'),
            'rpm_max':      ('RPM_Max', '{:.0f}'),
            'gear':         ('Gear', '{:.0f}'),
            'tps_mean':     ('TPS', '{:.1f}%'),
            'tps_max':      ('TPS_Max', '{:.1f}%'),
            'tps_delta':    ('TPS_Delta', '{:.1f}%'),
            'vss_mean':     ('Speed', '{:.2f}m/s'),
            'long_g_mean':  ('LongG', '{:.3f}'),
            'long_g_min':   ('LongG_Min', '{:.3f}'),
            'long_g_max':   ('LongG_Max', '{:.3f}'),
            'ign_angle':    ('Ign', '{:.1f}°'),
            'oil_temp':     ('Oil', '{:.1f}°C'),
            'coolant_temp': ('Coolant', '{:.1f}°C')
        }

        self.DOMAIN_QUESTIONS = {
            'engine': [
                "Audit 'Traction_Phase' segments: Identify any node where RPM dropped while TPS was > 85%. Is this a potential traction loss or engine bog?",
                "Look for 'TPS_Delta' spikes > 20% in the 'WOT' regime. Does the 'Ign_Angle' react immediately in the following node?",
                "Compare 'RPM_Max' in Gear 1 vs Gear 2 across the session. Are shifts occurring before or after the 7000 RPM threshold?"
            ],
            'braking': [
                "Analyze the transition from 'Trail_Braking' to 'Stationary_Idle'. Was the 'VSS_mean' reduction linear, or is there a 'LongG_Min' spike indicating a lock-up?",
                "Identify 'Hard_Braking' nodes where 'TPS' was > 5%. Is this a genuine 'Brake_Overlap' error or sensor noise in the Haltech log?",
                "What was the Speed (VSS) delta between the first and last node of the longest braking sequence?"
                "Verify a informed decision on where Rev Matching ranges are for this data?"
            ],
            'throttle': [
                "In 'Traction_Phase', identify nodes where 'TPS_mean' is between 25-75%. Is the driver 'hunting' for grip or maintaining a steady exit?",
                "Examine 'Maintenance_Throttle' nodes. Does the 'VSS_mean' increase, or is the car purely maintaining speed (within +/- 0.5 m/s)?",
                "Locate the highest 'TPS_Delta'. Cite the Node ID and describe the resulting change in 'LongG_mean' in the subsequent node."
            ],
            'thermal': [
                "Cross-reference 'Oil_Pressure' with 'LongG_Min'. In nodes where decel exceeds -1.5G, does pressure drop below the session mean?",
                "Trace 'Coolant_Temp' following 'WOT' segments. How many nodes does it take for the temperature to stabilize after the throttle is lifted?",
                "Identify the node with the highest 'Oil_Temp'. What was the 'Regime' and 'RPM' at that exact timestamp?"
            ],
            'default': [
                "Verify the STG 'Staircase': Are there any nodes where 'Gear' changes without a 'SHIFT' transition label?",
                "Check for 'Speed' (VSS) anomalies: Identify nodes where Speed increases despite a 'Lift_Off' or 'Hard_Braking' label."
            ]
        }

    def encode(self, rich_tokens, query):
        lines = []
        curr_session = None
        for t in rich_tokens:
            sess = t['session_id']
            if sess != curr_session:
                lines.append(f"\n--- SESSION {sess} ---")
                curr_session = sess

            m = t['metrics']
            start = t['time'][0]
            label = t['label'].split('|')[1] if '|' in str(t['label']) else str(t['label'])
            event_ctx = t['event_context']
            drv = t.get('driver', 'Normal')
            lam = m.get('lambda')
            transition_label = t.get('transition', 'Normal')
            if not transition_label or transition_label == "":
                transition_label = "Normal"
            lam_str = "N/A"
            
            if lam is not None or lam == 0:
                lam_str = "OFF/AIR"
                if lam > 0.5 and lam < 2.0:
                    #valid
                    if lam < 1.0:
                        lam_str = "RICH"
                    else:
                        lam_str = "LEAN"
                    
            flags = t.get('flags', [])
            flag_str = f" !{','.join(flags)}" if flags else ""
            #print("encode", m)
            
            clean_m = {}
            for key, val in m.items():
                if key in self.FORMAT_MAP:
                    label_name, fmt = self.FORMAT_MAP[key]
                    try:
                        # Cast to float to strip the np.float64, then format
                        clean_m[label_name] = fmt.format(float(val))
                    except (ValueError, TypeError):
                        clean_m[label_name] = str(val) # Fallback if data is weird
                else:
                    # If it's a metric not in the map, just cast it to float safely
                    try:
                        clean_m[key] = round(float(val), 2)
                    except (ValueError, TypeError):
                        clean_m[key] = str(val)

            # --- Final Append ---
            # Now we pass {clean_m} instead of {m}
            lines.append(
                f"[{start:.2f}] "
                f"Metrics:{clean_m} EVENT:{event_ctx} Driver:{drv} "
                f"Lambda: {lam_str} Flags: {flag_str} Transition:{t['transition']} Node:{t['id']} (<-{t['parent_id']})"
            )
        telemetry = "\n".join(lines)
        #print(telemetry)
        return f"\n{telemetry}\n"
    
    def cot_encode(self, rich_tokens, query, domain):
        base_questions = [
            "1. What was the highest RPM achieved? In which gear and regime?",
            "2. How many regime transitions occurred from WOT to Hard_Braking?",
            "3. What was the maximum speed achieved? At what timestamp?",
            "4. Which regime had the most windows? What percentage?",
            "5. Were there any gear shifts under high load (>7000 RPM)? List them."
        ]

        if isinstance(rich_tokens, list):
            parsed_nodes = rich_tokens
        else:
            parsed_nodes = self._parse_telemetry_string(rich_tokens)
        
        domain_key = (list(domain.keys())[0] if isinstance(domain, dict) else domain) or 'default'
        parsed_nodes = self._parse_telemetry_string(rich_tokens)
        summary = self._build_session_summary(parsed_nodes)
        specialized_questions = self.DOMAIN_QUESTIONS.get(domain_key, self.DOMAIN_QUESTIONS['default'])
        all_questions = base_questions + specialized_questions
        formatted_questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(all_questions)])
        telemetry = self._build_regime_grouped_telemetry(parsed_nodes, domain_key)
        gear_table = "### GEAR UTILIZATION TABLE\n" + self._generate_gear_table(parsed_nodes)
        prompt = (
            "### TASK: Analyze Telemetry DAG\n"
            "Analyze the following temporal nodes. Use Chain-of-Thought reasoning.\n"
            "STRUCTURE YOUR RESPONSE AS FOLLOWS:\n"
            "<thinking>\n"
            "1. Trace the STG edges (Transitions).\n"
            "2. Identify delta anomalies (e.g., RPM drops vs. Speed increases).\n"
            "3. Evaluate the 'Money Shift' (G4->G1) impact.\n"
            "</thinking>\n"
            "\n\n### TELEMETRY NODES\n"
            f"\n{telemetry}\n"
            "\n--- END TELEMETRY ---\n"
            f"\n{summary}\n"
            f"{gear_table}\n\n"
            "### MANDATORY ANALYSIS STEP\n"
            f"\n\n Analyse the <thinking> block with the data provided."
            f"Based on the telemetry above, answer these questions. "
            f"Cite specific timestamps and node IDs for each answer.\n\n"
            f"{formatted_questions}\n\n"
            f"USER QUESTION: {query}\n\n"
            f"Format: For each question, state the answer then cite evidence "
            f"Answer the User Question with the insights, if User Query answered, answer why the user query has already been answered."
            f"(timestamp, node ID, table row)."
            "\n### FINAL RESPONSE:\n"
        )
        return prompt

    def _parse_telemetry_string(self, text: str) -> List[Dict]:
        """Extracts stats from your specific f-string format using improved Regex."""
        nodes = []
        lines = text.strip().split('\n')
        
        for line in lines:
            if not line.startswith('['): 
                continue 
            
            # 1. Improved Regex: Supports negative numbers (-?) and different naming conventions
            time_match = re.search(r'\[([\d\.]+)\]', line)
            
            # Look for RPM or rpm_mean (optional in braking)
            rpm_match = re.search(r"'(?:RPM|rpm_mean)':\s*'(\d+)'", line)
            
            # Look for Gear
            gear_match = re.search(r"'Gear':\s*'(\d+)'", line)
            
            # Look for Speed or vss_mean (crucial for braking)
            speed_match = re.search(r"'(?:Speed|vss_mean)':\s*'([\d\.]+)m/s'", line)
            
            # Look for LongG (crucial for braking, supports negative values)
            g_match = re.search(r"'(?:long_g_min|long_g_mean)':\s*'(-?[\d\.]+)'", line)
            
            event_match = re.search(r'EVENT:([^\s]+)', line)
            node_match = re.search(r'Node:([\d_]+)', line)
            
            # 2. Relaxed Condition: Only require time and node_match to proceed
            if time_match and node_match:
                session_id = node_match.group(1).split('_')[0]
                
                raw_event = event_match.group(1) if event_match else "Unknown"
                regime = raw_event.split('|')[1] if '|' in raw_event else raw_event
                
                nodes.append({
                    'raw_line': line,
                    'time': float(time_match.group(1)),
                    'rpm': int(rpm_match.group(1)) if rpm_match else 0, # Default to 0 if missing
                    'gear': int(gear_match.group(1)) if gear_match else 0,
                    'speed': float(speed_match.group(1)) if speed_match else 0.0,
                    'long_g': float(g_match.group(1)) if g_match else 0.0,
                    'regime': regime,
                    'session_id': session_id
                })
        return nodes

    def _build_session_summary(self, nodes: List[Dict]) -> str:
        """Builds high-level stats from the parsed nodes."""
        if not nodes:
            return "=== SESSION SUMMARY ===\nNo valid nodes found.\n"
            
        lines = ["=== SESSION SUMMARY ===\n"]
        
        # Group by Session
        sessions = list(set(n['session_id'] for n in nodes))
        for sess in sorted(sessions):
            sess_nodes = [n for n in nodes if n['session_id'] == sess]
            
            rpms = [n['rpm'] for n in sess_nodes]
            gears = sorted(list(set(n['gear'] for n in sess_nodes if n['gear'] > 0)))
            speeds = [n['speed'] for n in sess_nodes]
            
            # Regime distribution
            regimes = {}
            for n in sess_nodes:
                regimes[n['regime']] = regimes.get(n['regime'], 0) + 1
            
            lines.append(f"Session {sess}:")
            lines.append(f"  Duration: {sess_nodes[0]['time']:.1f}s to {sess_nodes[-1]['time']:.1f}s ({len(sess_nodes)} STG nodes)")
            lines.append(f"  RPM range: {min(rpms)} - {max(rpms)}")
            lines.append(f"  Gears used: {gears if gears else [0]}")
            lines.append(f"  Max speed: {max(speeds):.1f} m/s")
            lines.append(f"  Regime distribution:")
            
            for regime, count in sorted(regimes.items(), key=lambda x: -x[1])[:5]:
                pct = count / len(sess_nodes) * 100
                lines.append(f"    {regime}: {count} ({pct:.0f}%)")
            lines.append("")
            
        return "\n".join(lines)

    def _build_regime_grouped_telemetry(self, nodes: List[Dict], domain: str) -> str:
        """Groups your raw f-strings by regime and samples them to save LLM context."""
        if domain == 'braking':
            nodes = [n for n in nodes if 'Braking' in n.get('regime', '')]
        lines = ["\n=== TELEMETRY BY REGIME ===\n"]
        
        sessions = list(set(n['session_id'] for n in nodes))
        for sess in sorted(sessions):
            lines.append(f"\n--- SESSION {sess} ---")
            sess_nodes = [n for n in nodes if n['session_id'] == sess]
            
            # Group by Regime
            by_regime = {}
            for n in sess_nodes:
                if n['regime'] not in by_regime:
                    by_regime[n['regime']] = []
                by_regime[n['regime']].append(n)
            
            # Sort regimes by size (or priority)
            for regime, reg_nodes in sorted(by_regime.items(), key=lambda x: -len(x[1])):
                lines.append(f"\n{regime} ({len(reg_nodes)} nodes):")
                
                # Sample the nodes (First 3, Middle 2, Last 3)
                sample = self._sample_nodes(reg_nodes)
                
                # Output the USER'S ORIGINAL f-string line
                for n in sample:
                    lines.append(f"  {n['raw_line']}")
                    
        return "\n".join(lines)

    def _sample_nodes(self, nodes: list, max_per_regime: int = 14) -> list:
        """
        Modified to provide a 20-30 node resolution.
        Takes the first 10, the last 10, and 10 distributed across the middle.
        """
        if len(nodes) <= max_per_regime:
            return nodes
            
        # Take the Head
        head = nodes[:4]
        
        # Take the Tail
        tail = nodes[-4:]
        
        # Sample the Body
        body_pool = nodes[4:-4]
        if body_pool:
            stride = max(1, len(body_pool) // 4)
            body = body_pool[::stride][:4]
        else:
            body = []
                
        # COMBINE (Ensure this is indented correctly!)
        combined = head + body + tail
        
        # Final Deduplication check
        sampled = []
        seen = set()
        for n in combined:
            if n['raw_line'] not in seen:
                seen.add(n['raw_line'])
                sampled.append(n)
                
        return sampled

    def _generate_gear_table(self, nodes: List[Dict]) -> str:
        """
        Creates a markdown table of gear utilization to 
        offload computation from the LLM.
        """
        if not nodes:
            return "No Gear Data Available."

        # 1. Aggregate stats
        stats = {}
        total_nodes = len(nodes)
        
        for n in nodes:
            g = n.get('gear', 0)
            rpm = n.get('rpm', 0)
            
            if g not in stats:
                stats[g] = {'count': 0, 'rpms': []}
            
            stats[g]['count'] += 1
            stats[g]['rpms'].append(rpm)

        # 2. Build Markdown Table
        table = [
            "| Gear | Node Count | % Usage | RPM Range (Min - Max) |",
            "|:----:|:----------:|:-------:|:---------------------:|"
        ]

        # Sort by Gear (0, 1, 2, etc.)
        for g in sorted(stats.keys()):
            count = stats[g]['count']
            pct = (count / total_nodes) * 100
            rpms = stats[g]['rpms']
            rpm_range = f"{min(rpms)} - {max(rpms)}"
            
            # Label Gear 0 as Neutral for the LLM
            gear_label = f"G{g} (N)" if g == 0 else f"G{g}"
            
            table.append(
                f"| {gear_label} | {count} | {pct:.1f}% | {rpm_range} |"
            )

        return "\n".join(table)
