import re
from typing import List, Dict, Any
from dataclasses import dataclass, field
from typing import Dict, List, Any
import uuid
import networkx as nx

class STGTokeniser:
    
    FIELD_SETS = {
        'engine':   ['rpm_mean', 'rpm_max', 'gear',
                    'tps_mean', 'tps_max','tps_delta',
                    'vss_mean', 'event_context', 'long_g_mean'],
        'braking':  ['vss_mean', 'rpm', 'rpm_mean', 'event_context', 'long_g_min', 'long_g_max', 
                     'tps_mean', 'gear'],
        'throttle': ['vss_mean', 'event_context', 'tps_mean', 'tps_max', 'tps_delta', 'gear',
                     'rpm_mean'],
        'thermal':  ['oil_temp', 'coolant_temp', 'oil_pressure', 
                     'vss_mean', 'event_context', 'gear', 'rpm'],
        'default':  ['rpm_mean', 'rpm_max', 'gear',
                     'vss_mean', 'event_context', 'oil_temp', 'coolant_temp', 'oil_pressure', 'long_g_mean', 'tps_mean', 'tps_delta', 
                     'long_g_min', 'long_g_max', 'ign_angle'],
    }

    def __init__(self, intent: dict = None):
        '''
        ST-LLM alignment: physics-informed node scoring - relevance to user query intent
        Intent instance from Spacy instance in __main__()
        '''
        self.intent = intent or {}
        self.FORMAT_MAP = {
            'RPM': ('RPM', '{:.0f}'),
            'Oil Temperature': ('Oil', '{:.1f}°C'),
            'Coolant Temperature': ('Coolant', '{:.1f}°C'),
            'Throttle Position': ('TPS', '{:.1f}%'),
            'Gear': ('Gear', '{:.0f}')
        }

    def _relevance_score(self, node) -> float:
        """
        Scores each node by how relevant it is to the current intent.
        Higher = more relevant. Replaces stride downsampling.
        This is the core ST-LLM token selection concept applied to
        your domain — query-aware, physics-grounded.
        """
        score = 0.0
        attrs   = node.attributes
        state   = attrs.get('state', '')
        session = str(attrs.get('session', ''))
        metrics = attrs.get('metrics', {})

        # Session match is highest-priority filter
        #if comparison == true? for target sessions in ?
        target_sessions = self.intent.get('sessions', [])
        if target_sessions:
            if session in target_sessions:
                score += 10.0
            else:
                return -1.0   # exclude entirely if wrong session

        # State relevance — does node label match intent states?
        target_states = self.intent.get('states', [])
        for ts in target_states:
            if ts.lower() in state.lower():
                score += 5.0

        # Anomaly nodes always surface — physically significant
        if attrs.get('flags'):
            score += 3.0

        # Dynamic regime nodes score higher than cruise/idle
        high_value_states = {
            'Hard_Braking', 'Trail_Braking', 'WOT',
            'Brake_Overlap', 'Aggressive_Transition'
        }
        if any(s in state for s in high_value_states):
            score += 2.0

        # Penalise stationary / idle — rarely useful in LLM context
        if any(s in state for s in ['Stationary', 'Idle', 'Stall']):
            score -= 2.0

        return score

    def _node_matches_intent(self, node) -> bool:
        """Hard filter — exclude nodes that cannot be relevant."""
        target_sessions = self.intent.get('sessions', [])
        if target_sessions:
            session = str(node.attributes.get('session', ''))
            if session not in target_sessions:
                return False
        return True

    def _project_fields(self, metrics: dict) -> dict:
        """
        Returns only the fields relevant to the current query domain.
        Keeps each token lean — ~15-25 tokens regardless of how rich
        the stored metrics dict is.
        """
        domain = self.intent.get('domain', 'default')
        fields = self.FIELD_SETS.get(domain, self.FIELD_SETS['default'])
        return {f: metrics[f] for f in fields if f in metrics}

    def tokenize_from_graph(self, stg: nx.DiGraph,
                         max_tokens: int = 10000) -> List[Dict]: #adjust max tokens here/see how to produce relevant results why starting from 4_191? - relevance score maybe, is it looking at relevance throughout the entire session nodes or just a slice of 100

        # Filter — NetworkX nodes are (node_id, attr_dict) tuples
        v_nodes = [
            (node_id, data)
            for node_id, data in stg.nodes(data=True)
            if self._node_matches_intent_nx(data)
        ]

        if not v_nodes:
            return []

        scored = []
        
        # Score by physics relevance (store with node_id, data, start_time)
        for node_id, data in v_nodes:
            score = self._relevance_score_nx(data)
            if score >= 0:   # only keep non‑excluded nodes
                start_time = data.get('start_time', 0)
                scored.append((score, node_id, data, start_time)) #store in scored

        if not scored:
            return []

        # Sort by start_time to get temporal order of scored array
        scored.sort(key=lambda x: x[3])   # sort by time

        # Determine number of bins (aim for ~2-5 nodes per bin)
        # Tune bin size and nodes per bin parameters
        num_bins = max(1, max_tokens // 5)   # e.g., 100 tokens -> 20 bins (5 per bin)
        nodes_per_bin = max(1, max_tokens // num_bins)

        # bin duration calculations
        total_duration = scored[-1][3] - scored[0][3]
        if total_duration <= 0:
            # Fallback if all times are identical
            bin_size = len(scored) // num_bins
        else:
            bin_size = total_duration / num_bins

        selected = [] # stores selected bins
        current_bin = 0
        while current_bin < num_bins and len(selected) < max_tokens: #max tokens defined in args parameters
            bin_start = scored[0][3] + current_bin * bin_size
            bin_end = bin_start + bin_size
            # Collect nodes in this time bin
            bin_nodes = [item for item in scored
                         if bin_start <= item[3] < bin_end]
            if bin_nodes:
                # Sort this bin by score descending, take top nodes_per_bin
                bin_nodes.sort(key=lambda x: x[0], reverse=True)
                selected.extend(bin_nodes[:nodes_per_bin])
            current_bin += 1

        # If max_tokens not met, fill with the next highest‑scored nodes
        if len(selected) < max_tokens and len(scored) > len(selected):
            remaining = [item for item in scored if item not in selected]
            remaining.sort(key=lambda x: x[0], reverse=True)
            selected.extend(remaining[:max_tokens - len(selected)])

        # Re‑sort selected chronologically for output
        selected.sort(key=lambda x: x[3])

        # Build token dicts
        rich_tokens = []
        prev_node_data = None # Tracks previous node to detect transitions
        last_node_id = "START"
        current_parent = "START"

        def fetch(metrics_dict, logic_name):
            """
            Maps a logical name (RPM, Gear, TPS) to the raw source keys 
            in your metrics dictionary and returns a float.
            """
            # Define the mapping from Logic Name to Source Key
            mapping = {
                'RPM': 'rpm_mean',
                'Gear': 'gear',
                'TPS': 'tps_mean'
            }
            
            source_key = mapping.get(logic_name)
            if not source_key:
                return 0.0
                
            val = metrics_dict.get(source_key, 0.0)
            
            # Handle 'None' strings or actual None types
            if val is None or str(val).lower() == 'none':
                return 0.0
                
            try:
                # Convert to float, stripping any units if they accidentally exists
                if isinstance(val, str):
                    val = val.replace('%', '').replace('rpm', '').strip()
                return float(val)
            except (ValueError, TypeError):
                return 0.0
        
    
        for i, (score, node_id, data, start_time) in enumerate(selected):
            metrics = data.get('metrics', {})
            m = data.get('metrics', {})
            projected = self._project_fields(metrics) # Filters only relevant engine fields
            transition = "Normal"
    
            # Apply Format Map to clean up floats and add units
            # This prevents "np.float64" strings from polluting Gemma's context
            formatted_metrics = {}
            for k, v in projected.items():
                if k in self.FORMAT_MAP: # self.FORMAT_MAP should be defined in your class
                    label, fmt = self.FORMAT_MAP[k]
                    try:
                        formatted_metrics[label] = fmt.format(float(v))
                    except (ValueError, TypeError):
                        formatted_metrics[label] = str(v)
                else:
                    formatted_metrics[k] = v

            preds = list(stg.predecessors(node_id))
            if preds:
                parent_id = preds[0]
                    
            if i == 0:
                transition = "TRACING_START"
                current_parent = "START"
            else:
                current_parent = last_node_id
                
                p_m = stg.nodes[last_node_id].get('metrics', {})
                
                curr_r = fetch(m, 'RPM')
                prev_r = fetch(p_m, 'RPM')
                
                curr_g = fetch(m, 'Gear')
                prev_g = fetch(p_m, 'Gear')
                
                curr_t = fetch(m, 'TPS')
                prev_t = fetch(p_m, 'TPS')

            # Edge Analysis (Only runs if a valid parent was found)
            def clean_val(v):
                if v is None or str(v).lower() == 'none': return 0.0
                return float(str(v).replace('%', '').replace('m/s', ''))
                
            def get_val(d, key):
                """Helper to extract and clean values from potential strings."""
                v = d.get(key, 0)
                if v is None or str(v).lower() == 'none': return 0.0
                # Remove units if they exist in the string
                if isinstance(v, str):
                    v = v.replace('%', '').replace('°C', '').replace('°', '').replace('m/s', '')
                try:
                    return float(v)
                except:
                    return 0.0

            raw_node_data = stg.nodes[node_id]
            metrics = raw_node_data.get('metrics', raw_node_data)

            if current_parent != "START":
                try:
                    p_node_raw = stg.nodes[current_parent]
                    p_metrics = p_node_raw.get('metrics', p_node_raw)
                
                    #print(f"DEBUG: ID {node_id} | RPM extracted: {curr_r}")
                    #print(f"ID: {node_id} | G: {prev_g}->{curr_g} | T: {prev_t}->{curr_t} | R: {prev_r}->{curr_r}")

                    # Check for specific engine events
                    if int(curr_g) != int(prev_g):
                        transition = f"SHIFT: G{int(prev_g)}->G{int(curr_g)}"
                    elif curr_t > 80.0 and prev_t < 30.0:
                        transition = "WOT_ACCEL"
                    elif (curr_r - prev_r) > 1500:
                        transition = "POWER_SURGE"
                    elif (prev_r - curr_r) > 1000:
                        transition = "ENGINE_BRAKING"
                    # GAP JUMP (Visualizes the downsampling for the LLM)
                    try:
                        # Ensure both IDs follow the 'session_index' format
                        if '_' in node_id and '_' in current_parent:
                            curr_idx = int(node_id.split('_')[-1])
                            prev_idx = int(current_parent.split('_')[-1])
                            
                            if (curr_idx - prev_idx) > 1:
                                gap_str = f"GAP({curr_idx - prev_idx} nodes)"
                                transition = gap_str if transition == "Normal" else f"{transition} | {gap_str}"
                    except (ValueError, IndexError):
                        pass
        
                except Exception as e:
                    transition = "STABLE_DATA"

            last_node_id = node_id
            
            #print(self._project_fields(metrics))
            #print(data.get('event_context'))
            #print(data.get('composite', data.get('event_context', 'unknown')))
            #print("fm:", formatted_metrics)
            rich_tokens.append({
            "id":             node_id,
            "parent_id":      last_node_id, # Linkage for Spatio-Temporal awareness
            "time":           (data.get('start_time'), data.get('end_time')),
            "session_id":     data.get('session_id'),
            "event_context":  data.get('composite', data.get('event_context', 'unknown')),
            "label":          data.get('composite', data.get('long_regime', 'unknown')),
            "metrics":        formatted_metrics, # Cleaned/Formatted dictionary
            "spatial":        data.get('lap_dist', 0),
            "flags":          data.get('flags', []),
            "driver":         self._get_behavior_nx(node_id, stg),
            "transition":     transition # Explicit label for shifts/major changes
        })
            
        prev_node_data = metrics

        print(f"STGTokeniser: {len(v_nodes)} nodes → {len(rich_tokens)} selected "
          f"(domain={self.intent.get('domain','default')})")
        return rich_tokens


    def _node_matches_intent_nx(self, data: dict) -> bool:
        target_sessions = self.intent.get('sessions', [])
        if target_sessions:
            if str(data.get('session_id')) not in target_sessions:
                return False
        return True
    
    def _relevance_score_nx(self, data: dict) -> float:
        score       = 0.0
        session     = str(data.get('session_id', ''))
        long_regime = data.get('long_regime', '')
        flags       = data.get('flags', [])
        domain      = self.intent.get('domain', 'default')
        metrics     = data.get('metrics', {})
        target_sessions = self.intent.get('sessions', [])
    
        if target_sessions:
            if session not in target_sessions:
                return -1.0
            score += 10.0
    
        if flags:
            score += 4.0
    
        if domain in ('throttle', 'engine'):
            # Prioritise high-value throttle regimes
            high_info = {'WOT', 'Partial_Throttle', 'Brake_Overlap'}
            if long_regime in high_info:
                score += 5.0
    
            # Penalise heavily — these add no throttle/RPM insight
            if long_regime in ('Stationary', 'Idle', 'Stall',
                               'Lift_Off', 'Coasting'):
                score -= 8.0
    
            # Reward high RPM — indicates active engine load
            rpm = metrics.get('rpm_mean', 0)
            if rpm and rpm > 6000:
                score += 3.0
            elif rpm and rpm > 4000:
                score += 1.0
    
            # Reward high TPS — confirms throttle intent
            tps = metrics.get('tps_mean', 0)
            if tps and tps > 70:
                score += 3.0
            elif tps and tps > 40:
                score += 1.0
    
            # Reward valid lambda — sensor active
            lam = metrics.get('lambda')
            if lam is not None:
                score += 4.0          # rare — surface these always
                if lam > 1.05:
                    score += 2.0      # lean — engineering concern
                elif lam < 0.88:
                    score += 2.0      # rich — also concern
    
            rpm_flag = data.get('rpm_flag', '')
            if rpm_flag in ('Near_Limiter', 'Lugging'):
                score += 3.0
    
        elif domain == 'braking':
            if long_regime in ('Hard_Braking', 'Trail_Braking'):
                score += 5.0
            long_g = metrics.get('long_g_min', 0)
            if long_g and long_g < -0.8:
                score += 2.0
    
        elif domain == 'thermal':
            if flags:
                score += 6.0
            oil_t = metrics.get('oil_temp')
            if oil_t and oil_t > 100:
                score += 3.0
    
        # Penalise stationary regardless of domain
        if long_regime in ('Stationary', 'Idle', 'Stall'):
            score -= 5.0
    
        return score

        
    def _get_behavior_nx(self, node_id, G: nx.DiGraph) -> str:
        for u, v, data in G.in_edges(node_id, data=True):
            if data.get('edge_type') == 'transition':
                return data.get('from_state', 'Normal')
        return 'Normal'

