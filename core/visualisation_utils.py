import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

class Visualisation_Tools:
    def __init__(self):
        # Colour map for long_regime labels
        self.REGIME_COLOURS = {
            'Hard_Braking':          '#E24B4A',
            'Trail_Braking':         '#F09595',
            'Brake_Overlap':         '#EF9F27',
            'Coasting':              '#B5D4F4',
            'WOT':                   '#1D9E75',
            'Traction_Limited':          '#9FE1CB',
            'Partial_Throttle':      '#639922',
            'Maintenance_Throttle':  '#C0DD97',
            'Aggressive_Transition': '#7F77DD',
            'Cruise/Steady':         '#D3D1C7',
            'Lift_Off':              '#888780', 
            'Holding_Throttle':      '#678780',
            'Transition_Phase':      '#676767',
            'Coasting':              '#696969',
            'Stationary_Idle':       '#101010',
        }
        self.DEFAULT_COLOUR = '#888780'
    '''Dedicated to Final Visualisations'''
    
    
    def plot_graph_timeline(self, G: nx.DiGraph, session_id: str = None,
                             max_nodes: int = 10000):
        '''Plot event timeline -> View trace of event plotted against time'''
        nodes = [
            (n, d) for n, d in G.nodes(data=True)
            if (session_id is None or str(d.get('session_id')) == str(session_id))
        ]
        nodes = nodes[:max_nodes]
        if not nodes:
            print("No nodes found for that session.")
            return
    
        nodes.sort(key=lambda x: x[1].get('start_time', 0))
    
        regimes    = sorted(set(REGIME_COLOURS.keys()))
        regime_idx = {r: i for i, r in enumerate(regimes)}
    
        # set axes
        xs      = [d.get('start_time', i) for i, (_, d) in enumerate(nodes)]
        ys      = [regime_idx.get(d.get('long_regime', ''), 0) for _, d in nodes]
        colours = [self.REGIME_COLOURS.get(d.get('long_regime', ''), self.DEFAULT_COLOUR)
                   for _, d in nodes]
    
        fig, axes = plt.subplots(2, 1, figsize=(16, 8),
                                  gridspec_kw={'height_ratios': [3, 1]})
        fig.suptitle(f"STG Timeline — session {session_id or 'all'}",
                     fontsize=13, fontweight='normal')
    
        ax = axes[0]
        ax.scatter(xs, ys, c=colours, s=18, alpha=0.85, linewidths=0)
    
        node_map = {n: (xs[i], ys[i]) for i, (n, _) in enumerate(nodes)}
        for u, v, ed in G.edges(data=True):
            if ed.get('edge_type') == 'transition':
                if u in node_map and v in node_map:
                    x0, y0 = node_map[u]
                    x1, y1 = node_map[v]
                    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                                arrowprops=dict(arrowstyle="-|>",
                                               color='#888780',
                                               lw=0.4, alpha=0.4))
    
        ax.set_yticks(list(regime_idx.values()))
        ax.set_yticklabels(list(regime_idx.keys()), fontsize=8)
        ax.set_xlabel("Session time (seconds)")   # ── Fixed label
        ax.set_ylabel("Longitudinal regime")
        ax.grid(axis='x', alpha=0.2)
    
        patches = [mpatches.Patch(color=c, label=r)
                   for r, c in self.REGIME_COLOURS.items()]
        ax.legend(handles=patches, fontsize=7, ncol=3,
                  loc='upper right', framealpha=0.7)
    
        ax2 = axes[1]
        vss_vals = [d.get('metrics', {}).get('vss_mean', 0) for _, d in nodes]
        ax2.plot(xs, vss_vals, color='#378ADD', linewidth=0.8, alpha=0.9)
        ax2.set_ylabel("Speed (m/s)", fontsize=8)
        ax2.set_xlabel("Session time (seconds)")  # ── Fixed label
        ax2.grid(alpha=0.2)
    
        plt.tight_layout()
        plt.savefig(f"stg_timeline_s{session_id or 'all'}.png",
                    dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Saved: stg_timeline_s{session_id or 'all'}.png")
    
    
    def plot_graph_network(self, G: nx.DiGraph, session_id: str = None,
                            max_nodes: int = 10000):
        """
        Network topology view — shows edges between regimes.
        Use this for dissertation methodology figures, not analysis.
        """
        nodes_to_show = [
            n for n, d in G.nodes(data=True)
            if (session_id is None or str(d.get('session_id')) == str(session_id))
        ][:max_nodes]
    
        subG = G.subgraph(nodes_to_show)
    
        colours = [
            REGIME_COLOURS.get(subG.nodes[n].get('long_regime', ''), self.DEFAULT_COLOUR)
            for n in subG.nodes
        ]
    
        # Spring layout — groups similar states loosely
        pos = nx.spring_layout(subG, seed=42, k=1.2)
    
        fig, ax = plt.subplots(figsize=(14, 8))
    
        nx.draw_networkx_nodes(subG, pos, node_color=colours,
                               node_size=60, alpha=0.9, ax=ax)
        nx.draw_networkx_edges(subG, pos, alpha=0.15,
                               edge_color='#888780',
                               arrows=True, arrowsize=8, ax=ax)
    
        # Only label transition edges
        transition_edges = {(u, v): d.get('to_state', '')[:6]
                            for u, v, d in subG.edges(data=True)
                            if d.get('edge_type') == 'transition'}
        nx.draw_networkx_edge_labels(subG, pos, transition_edges,
                                      font_size=6, alpha=0.6, ax=ax)
    
        patches = [mpatches.Patch(color=c, label=r)
                   for r, c in self.REGIME_COLOURS.items()
                   if c in colours]
        ax.legend(handles=patches, fontsize=7, loc='upper left')
        ax.set_title(f"STG network — session {session_id or 'all'} "
                     f"(first {max_nodes} nodes)", fontsize=11)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"stg_network_s{session_id or 'all'}.png",
                    dpi=150, bbox_inches='tight')
        plt.show()