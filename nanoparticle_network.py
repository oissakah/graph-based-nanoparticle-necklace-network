"""
Nanoparticle Network Circuit Model

Models nanoparticle networks as graphs where:
- Nodes represent junctions between nanoparticles
- Edges represent nanoparticle arrangements with electrical resistance
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

class NanoparticleNetwork:
    def __init__(self, n_junctions=50, connection_radius=0.2, domain=(1.0, 1.0)):
        """
        Initialize nanoparticle network

        Parameters:
        -----------
        n_junctions : int
            Number of junctions (nodes) in the network
        connection_radius : float
            Maximum distance for connecting junctions (0-1 normalized units)
        domain : tuple
            (width, height) of the network domain
        """
        self.G = nx.Graph()
        self.n_junctions = n_junctions
        self.connection_radius = connection_radius
        self.domain = domain
        self.positions = None
        self.source_nodes = []  # Multiple source nodes on the left
        self.drain_nodes = []   # Multiple drain nodes on the right
        self.source_node = None  # Backward compatibility
        self.drain_node = None   # Backward compatibility

    def generate_network(self, seed=None, node_Vth=(0.1, 0.3, 0.5), edge_k=1000.0, node_r_scale=0.0, r_floor=0.0):
        """
        Generate spatially-distributed random network

        Parameters:
        -----------
        seed : int, optional
            Random seed for reproducibility
        node_Vth : tuple, list, or dict
            Threshold voltage configuration for nodes. Can be:
            - tuple/list: Discrete values to randomly choose from, e.g., (4.0, 8.0, 12.0)
            - dict: Distribution specification with 'type' and parameters:
                - {'type': 'uniform', 'min': 2.0, 'max': 15.0}
                - {'type': 'normal', 'mean': 8.0, 'std': 3.0, 'min': 0.0}
                - {'type': 'lognormal', 'mean': 2.0, 'sigma': 0.5}
                - {'type': 'exponential', 'scale': 5.0, 'offset': 2.0}
                - {'type': 'gamma', 'shape': 2.0, 'scale': 3.0, 'offset': 0.0}
                - {'type': 'weibull', 'shape': 2.0, 'scale': 8.0}
        edge_k : float
            Proportionality constant for edge resistance (R_edge = k * distance)
        node_r_scale : float
            Proportionality constant for node resistance (R_node = node_r_scale * Vth).
            Default 0.0 = edge-only model (existing behavior).
            Set e.g. 1e9 to enable junction-node resistance model.
        """
        if seed is not None:
            np.random.seed(seed)

        width, height = self.domain
        self.positions = np.random.rand(self.n_junctions, 2) * [width, height]

        # Determine if using distribution or discrete values
        use_distribution = isinstance(node_Vth, dict)

        for i in range(self.n_junctions):
            if use_distribution:
                Vth = self._sample_Vth_distribution(node_Vth)
            else:
                # Randomly choose from discrete Vth values (multimodal distribution)
                Vth = float(np.random.choice(node_Vth))
            self.G.add_node(i,
                          pos=self.positions[i],
                          Vth=Vth,
                          # R_node uses a floored Vth (max(Vth, r_floor)) so that
                          # low activation voltages don't collapse node resistance
                          # toward zero. Activation still uses the true Vth above;
                          # only the resistance is floored. r_floor=0.0 reproduces
                          # the original unfloored behavior.
                          R_node=node_r_scale * max(Vth, r_floor),
                          activated=False)  # Track if node is activated

        # Connect junctions within connection radius
        # Edge resistance proportional to distance (Au nanoparticle arrangement)
        for i in range(self.n_junctions):
            for j in range(i+1, self.n_junctions):
                dist = np.linalg.norm(self.positions[i] - self.positions[j])
                if dist < self.connection_radius:
                    R_edge = edge_k * dist  # Linear with distance
                    self.G.add_edge(i, j,
                                  resistance=R_edge,
                                  distance=dist,
                                  length=dist,
                                  R_edge=R_edge,
                                  activated=False)  # Track if edge is activated

        # Junctions must connect at least 2 nanoparticles
        # Iterate until no more nodes to remove (removing nodes can create new degree-1 nodes)
        total_removed = 0
        while True:
            nodes_to_remove = [node for node, degree in self.G.degree() if degree < 2]
            if not nodes_to_remove:
                break
            self.G.remove_nodes_from(nodes_to_remove)
            total_removed += len(nodes_to_remove)

        if total_removed > 0:
            remaining_nodes = list(self.G.nodes())
            self.positions = self.positions[remaining_nodes]
            mapping = {old_node: new_node for new_node, old_node in enumerate(remaining_nodes)}
            self.G = nx.relabel_nodes(self.G, mapping)
            self.n_junctions = self.G.number_of_nodes()
            print(f"Removed {total_removed} nodes with degree < 2 (not true junctions)")

        print(f"Generated network: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")

    def _sample_Vth_distribution(self, dist_config):
        """
        Sample a single Vth value from the specified distribution.

        Parameters:
        -----------
        dist_config : dict
            Distribution configuration with 'type' and parameters

        Returns:
        --------
        float
            Sampled Vth value
        """
        dist_type = dist_config.get('type', 'uniform')

        if dist_type == 'uniform':
            vmin = dist_config.get('min', 0.0)
            vmax = dist_config.get('max', 10.0)
            return np.random.uniform(vmin, vmax)

        elif dist_type == 'normal' or dist_type == 'gaussian':
            mean = dist_config.get('mean', 8.0)
            std = dist_config.get('std', 2.0)
            vmin = dist_config.get('min', 0.0)  # Clip to avoid negative Vth
            vmax = dist_config.get('max', np.inf)
            value = np.random.normal(mean, std)
            return float(np.clip(value, vmin, vmax))

        elif dist_type == 'lognormal':
            # Parameters are for the underlying normal distribution
            mean = dist_config.get('mean', 2.0)  # mean of ln(X)
            sigma = dist_config.get('sigma', 0.5)  # std of ln(X)
            return float(np.random.lognormal(mean, sigma))

        elif dist_type == 'exponential':
            scale = dist_config.get('scale', 5.0)  # mean of exponential
            offset = dist_config.get('offset', 0.0)  # shift the distribution
            return offset + np.random.exponential(scale)

        elif dist_type == 'gamma':
            shape = dist_config.get('shape', 2.0)  # k parameter
            scale = dist_config.get('scale', 3.0)  # theta parameter
            offset = dist_config.get('offset', 0.0)
            return offset + np.random.gamma(shape, scale)

        elif dist_type == 'weibull':
            shape = dist_config.get('shape', 2.0)  # k parameter
            scale = dist_config.get('scale', 8.0)  # lambda parameter
            return scale * np.random.weibull(shape)
        
        elif dist_type == 'bimodal':
            # Mixture of two distributions
            weight = dist_config.get('weight', 0.5)
            mode1 = dist_config.get('mode1')
            mode2 = dist_config.get('mode2')
    
            if mode1 is None or mode2 is None:
                raise ValueError("Bimodal distribution requires 'mode1' and 'mode2'.")
    
            if np.random.rand() < weight:
                return self._sample_Vth_distribution(mode1)
            else:
                return self._sample_Vth_distribution(mode2)

        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")

    def set_electrodes(self, source=0, drain=None):
        """
        Set source and drain electrodes (single node mode - backward compatibility)

        Parameters:
        -----------
        source : int
            Source node index
        drain : int, optional
            Drain node index (defaults to last node)
        """
        self.source_node = source
        self.drain_node = drain if drain is not None else self.n_junctions - 1

    def identify_sources_drains(self, left_thresh=0.15, right_thresh=0.85):
        """
        Identify source nodes (left side) and drain nodes (right side) based on x-position

        Parameters:
        -----------
        left_thresh : float
            x-position threshold for source nodes (nodes with x <= left_thresh * domain_width)
        right_thresh : float
            x-position threshold for drain nodes (nodes with x >= right_thresh * domain_width)

        Returns:
        --------
        tuple
            (sources, drains) - lists of node IDs
        """
        width = self.domain[0]
        sources = []
        drains = []

        for node in self.G.nodes():
            x_pos = self.positions[node][0]
            if x_pos <= left_thresh * width:
                sources.append(node)
            elif x_pos >= right_thresh * width:
                drains.append(node)

        self.source_nodes = sources
        self.drain_nodes = drains

        print(f"Identified {len(sources)} source nodes and {len(drains)} drain nodes")
        return sources, drains

    def save_network(self, filename):
        """Save network to file using pickle"""
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.G, f)
        np.save(filename.replace('.pkl', '_positions.npy'), self.positions)
        print(f"Network saved to {filename}")

    def load_network(self, filename):
        """Load network from file using pickle"""
        import pickle
        with open(filename, 'rb') as f:
            self.G = pickle.load(f)
        self.positions = np.load(filename.replace('.pkl', '_positions.npy'))
        self.n_junctions = self.G.number_of_nodes()
        print(f"Network loaded from {filename}")

    def _calculate_path_current(self, path, voltage, tunneling_effects=True):
        """
        Calculate total resistance and current through the percolation path.

        The path consists of nodes (junctions) and edges.
        Total resistance = sum of edge resistances along the path.

        The effective voltage drop available for current is:
        V_effective = V_applied - V_threshold
        where V_threshold is the common threshold voltage of all nodes in the path
        (all nodes in an activated path have Vth <= V_applied, we use the maximum Vth
        in the path as the effective barrier voltage).

        Current = V_effective / R_total

        Parameters:
        -----------
        path : list
            List of node IDs in the percolation path
        voltage : float
            Applied voltage

        Returns:
        --------
        tuple
            (total_resistance, current) in Ohms and Amperes
        """
        total_resistance = 0.0

        # Sum node resistances (0.0 in edge-only mode)
        for node in path:
            total_resistance += self.G.nodes[node].get('R_node', 0.0)

        # Sum edge resistances along the path
        for i in range(len(path) - 1):
            node1, node2 = path[i], path[i+1]
            edge_data = self.G[node1][node2]
            total_resistance += edge_data['R_edge']

        if tunneling_effects:
        # Find the maximum threshold voltage in the path (the limiting barrier)
            path_Vth_max = max(self.G.nodes[node]['Vth'] for node in path)
        else:
            path_Vth_max = 0.0

        # Effective voltage is applied voltage minus threshold barrier
        ##############################################################
        V_effective = voltage - path_Vth_max
        ##############################################################

        if total_resistance > 0 and V_effective > 0:
            current = V_effective / total_resistance
        else:
            current = 0.0

        return total_resistance, current

    def _save_voltage_sweep_image(self, voltage, activated_nodes, activated_edges,
                                   percolation_path, image_dir, step_num):
        """
        Save an image of the network at the current voltage step.

        Parameters:
        -----------
        voltage : float
            Current voltage
        activated_nodes : set
            Set of activated node IDs
        activated_edges : set
            Set of activated edge tuples
        percolation_path : list or None
            Percolation path if found
        image_dir : str
            Directory to save images
        step_num : int
            Step number for filename
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        pos = self.positions

        for i, j in self.G.edges():
            ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
                   'lightgray', alpha=0.2, linewidth=0.5, zorder=1)

        for edge in activated_edges:
            i, j = edge
            ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
                   'blue', alpha=0.6, linewidth=1.5, zorder=2)

        if percolation_path is not None:
            for k in range(len(percolation_path) - 1):
                i, j = percolation_path[k], percolation_path[k+1]
                ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
                       'red', alpha=0.9, linewidth=3, zorder=4)

        all_nodes = list(self.G.nodes())
        ax.scatter(pos[all_nodes, 0], pos[all_nodes, 1],
                  c='lightgray', s=20, zorder=3, edgecolors='black', linewidth=0.3)

        if activated_nodes:
            activated_list = list(activated_nodes)
            ax.scatter(pos[activated_list, 0], pos[activated_list, 1],
                      c='lightblue', s=40, zorder=4, edgecolors='blue', linewidth=0.8)

        # Highlight path nodes if percolation found
        if percolation_path is not None:
            path_pos = pos[percolation_path]
            ax.scatter(path_pos[:, 0], path_pos[:, 1],
                      c='orange', s=80, zorder=5, edgecolors='red', linewidth=1.5)

        if self.source_nodes:
            source_pos = pos[self.source_nodes]
            ax.scatter(source_pos[:, 0], source_pos[:, 1],
                      c='green', s=120, marker='s', label='Sources', zorder=6,
                      edgecolors='darkgreen', linewidth=2)

        if self.drain_nodes:
            drain_pos = pos[self.drain_nodes]
            ax.scatter(drain_pos[:, 0], drain_pos[:, 1],
                      c='red', s=120, marker='s', label='Drains', zorder=6,
                      edgecolors='darkred', linewidth=2)

        if percolation_path is not None:
            ax.set_title(f'V = {voltage:.3f}V - PERCOLATION FOUND!\n' +
                        f'{len(activated_nodes)} nodes activated, Path length: {len(percolation_path)}',
                        fontsize=12, fontweight='bold', color='red')
        else:
            ax.set_title(f'V = {voltage:.3f}V\n{len(activated_nodes)} nodes activated',
                        fontsize=12)

        ax.set_xlabel('x position', fontsize=10)
        ax.set_ylabel('y position', fontsize=10)
        ax.set_aspect('equal')
        ax.legend(fontsize=9)
        plt.tight_layout()

        filename = f"{image_dir}/step_{step_num:04d}_V_{voltage:.3f}V.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)

        if step_num % 10 == 0:
            print(f"  Saved image: {filename}")

    def find_all_percolation_paths(self, activated_nodes, activated_edges, max_paths=None):
        """
        Find all independent percolation paths from sources to drains.

        Uses BFS to find node-disjoint paths (except at sources and drains).
        This represents parallel conduction pathways through the network.

        Parameters:
        -----------
        activated_nodes : set
            Set of activated node IDs
        activated_edges : set
            Set of activated edge tuples
        max_paths : int or None
            Maximum number of paths to find (None = find all)

        Returns:
        --------
        list of lists
            List of paths, where each path is a list of node IDs
        """
        from collections import deque

        paths = []
        used_nodes = set()  # Nodes already used in paths (except sources/drains)

        iteration = 0
        while max_paths is None or len(paths) < max_paths:
            iteration += 1
            if iteration > 1000:  # Safety limit
                break

            # BFS from sources to drains, avoiding used nodes
            queue = deque()
            visited = set()
            parent = {}

            for source in self.source_nodes:
                if source in activated_nodes:
                    queue.append(source)
                    visited.add(source)
                    parent[source] = None

            path_found = False
            while queue:
                current = queue.popleft()

                if current in self.drain_nodes:
                    # Reconstruct path
                    path = []
                    node = current
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path = path[::-1]  # Reverse to get source -> drain

                    for node in path:
                        if node not in self.source_nodes and node not in self.drain_nodes:
                            used_nodes.add(node)

                    paths.append(path)
                    path_found = True
                    break

                # Explore neighbors through activated edges
                for neighbor in self.G.neighbors(current):
                    # Skip if already visited or if it's a used intermediate node
                    if neighbor in visited:
                        continue
                    if neighbor in used_nodes:
                        continue
                    if neighbor not in activated_nodes:
                        continue

                    edge = (current, neighbor) if (current, neighbor) in activated_edges else (neighbor, current)
                    if edge in activated_edges:
                        visited.add(neighbor)
                        parent[neighbor] = current
                        queue.append(neighbor)

            if not path_found:
                break  # No more paths found

        return paths

    def analyze_path_complexity(self, result=None):
        """
        Analyze the complexity of the percolation network and paths.

        Quantifies path-finding complexity with various metrics including
        network density, path diversity, and independence of parallel paths.

        Parameters:
        -----------
        result : dict, optional
            Result from voltage_sweep_full_percolation() to analyze

        Returns:
        --------
        dict with metrics
        """
        import networkx as nx

        if not self.source_nodes or not self.drain_nodes:
            raise ValueError("Must identify sources and drains first")

        metrics = {'network_metrics': {}, 'path_metrics': {}}

        # Network-level metrics
        n_nodes = self.G.number_of_nodes()
        n_edges = self.G.number_of_edges()
        metrics['network_metrics']['total_nodes'] = n_nodes
        metrics['network_metrics']['total_edges'] = n_edges

        degrees = [d for n, d in self.G.degree()]
        avg_degree = sum(degrees) / len(degrees) if degrees else 0
        metrics['network_metrics']['avg_degree'] = avg_degree

        max_edges = n_nodes * (n_nodes - 1) / 2
        density = n_edges / max_edges if max_edges > 0 else 0
        metrics['network_metrics']['network_density'] = density

        try:
            clustering = nx.average_clustering(self.G)
            metrics['network_metrics']['clustering_coefficient'] = clustering
        except:
            metrics['network_metrics']['clustering_coefficient'] = 0.0

        # Path-level metrics
        n_sources = len(self.source_nodes)
        n_drains = len(self.drain_nodes)
        metrics['path_metrics']['num_sources'] = n_sources
        metrics['path_metrics']['num_drains'] = n_drains

        max_possible_paths = n_sources * n_drains * (avg_degree ** (n_nodes // 10))
        if max_possible_paths > 1e10:
            metrics['path_metrics']['max_possible_simple_paths'] = 'very large (>10^10)'
        else:
            metrics['path_metrics']['max_possible_simple_paths'] = int(max_possible_paths)

        # Analyze found paths
        if result and 'all_paths' in result:
            paths = result['all_paths']
            n_paths = len(paths)
            metrics['path_metrics']['paths_found'] = n_paths

            max_pairs = n_sources * n_drains
            metrics['path_metrics']['path_efficiency'] = n_paths / max_pairs if max_pairs > 0 else 0

            if n_paths > 0:
                path_lengths = [len(p) for p in paths]
                metrics['path_metrics']['avg_path_length'] = sum(path_lengths) / len(path_lengths)
                metrics['path_metrics']['path_length_std'] = (
                    sum((l - metrics['path_metrics']['avg_path_length'])**2 for l in path_lengths) / len(path_lengths)
                ) ** 0.5
                metrics['path_metrics']['shortest_path_length'] = min(path_lengths)
                metrics['path_metrics']['longest_path_length'] = max(path_lengths)

                all_path_nodes = set()
                for path in paths:
                    all_path_nodes.update(path)

                intermediate_nodes = all_path_nodes - set(self.source_nodes) - set(self.drain_nodes)
                total_intermediate = n_nodes - n_sources - n_drains
                diversity = len(intermediate_nodes) / total_intermediate if total_intermediate > 0 else 0
                metrics['path_metrics']['path_diversity_score'] = diversity

                if n_paths > 1:
                    overlaps = []
                    for i in range(n_paths):
                        for j in range(i+1, n_paths):
                            path_i_set = set(paths[i]) - set(self.source_nodes) - set(self.drain_nodes)
                            path_j_set = set(paths[j]) - set(self.source_nodes) - set(self.drain_nodes)
                            if path_i_set and path_j_set:
                                overlap = len(path_i_set & path_j_set) / min(len(path_i_set), len(path_j_set))
                                overlaps.append(overlap)

                    avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
                    metrics['path_metrics']['avg_path_overlap'] = avg_overlap
                    metrics['path_metrics']['path_independence'] = 1.0 - avg_overlap
                else:
                    metrics['path_metrics']['avg_path_overlap'] = 0.0
                    metrics['path_metrics']['path_independence'] = 1.0

        # Overall complexity score
        complexity_components = []
        complexity_components.append(min(density * 5, 1.0))
        complexity_components.append(metrics['network_metrics']['clustering_coefficient'])
        complexity_components.append(min(avg_degree / 10.0, 1.0))
        if 'path_diversity_score' in metrics['path_metrics']:
            complexity_components.append(metrics['path_metrics']['path_diversity_score'])

        metrics['complexity_score'] = sum(complexity_components) / len(complexity_components)
        return metrics

    def print_complexity_analysis(self, result=None):
        """Print formatted complexity analysis report."""
        metrics = self.analyze_path_complexity(result)

        print("\n" + "="*70)
        print("NETWORK COMPLEXITY ANALYSIS")
        print("="*70)

        print("\n[Network Structure]")
        nm = metrics['network_metrics']
        print(f"  Total nodes:              {nm['total_nodes']}")
        print(f"  Total edges:              {nm['total_edges']}")
        print(f"  Average degree:           {nm['avg_degree']:.2f}")
        print(f"  Network density:          {nm['network_density']:.4f} ({nm['network_density']*100:.2f}%)")
        print(f"  Clustering coefficient:   {nm['clustering_coefficient']:.4f}")

        print("\n[Path Complexity]")
        pm = metrics['path_metrics']
        print(f"  Source nodes:             {pm['num_sources']}")
        print(f"  Drain nodes:              {pm['num_drains']}")
        print(f"  Max source-drain pairs:   {pm['num_sources'] * pm['num_drains']}")
        print(f"  Est. max simple paths:    {pm['max_possible_simple_paths']}")

        if 'paths_found' in pm:
            print(f"\n[Actual Paths Found]")
            print(f"  Number of paths found:    {pm['paths_found']}")
            print(f"  Path efficiency:          {pm['path_efficiency']:.4f} ({pm['path_efficiency']*100:.2f}%)")

            if pm['paths_found'] > 0:
                print(f"  Shortest path length:     {pm['shortest_path_length']} nodes")
                print(f"  Longest path length:      {pm['longest_path_length']} nodes")
                print(f"  Average path length:      {pm['avg_path_length']:.2f} nodes")
                print(f"  Path length std dev:      {pm['path_length_std']:.2f}")
                print(f"  Path diversity score:     {pm['path_diversity_score']:.4f} ({pm['path_diversity_score']*100:.2f}%)")

                if 'path_independence' in pm:
                    print(f"  Path independence:        {pm['path_independence']:.4f} ({pm['path_independence']*100:.2f}%)")
                    print(f"  Avg path overlap:         {pm['avg_path_overlap']:.4f}")

        print(f"\n[Overall Complexity Score]")
        print(f"  Complexity score:         {metrics['complexity_score']:.4f} / 1.00")

        if metrics['complexity_score'] < 0.3:
            complexity_level = "LOW - Simple network with few paths"
        elif metrics['complexity_score'] < 0.6:
            complexity_level = "MEDIUM - Moderate network complexity"
        else:
            complexity_level = "HIGH - Complex network with many alternative paths"

        print(f"  Interpretation:           {complexity_level}")
        print("="*70)

        return metrics

    def calculate_iv_curve(self, V_start=0.0, V_max=2.0, V_step=0.01, tunneling_effects=True):
        """
        Calculate I-V curve by sweeping voltage and computing total current.

        For each applied voltage:
        1. Activate nodes where Vth <= V_applied
        2. Find all percolation paths through activated nodes
        3. Calculate current through each path: I = (V - Vth_max_in_path) / R_path
        4. Sum currents from all parallel paths to get total current

        Parameters:
        -----------
        V_start : float
            Starting voltage
        V_max : float
            Maximum voltage
        V_step : float
            Voltage increment

        Returns:
        --------
        dict
            {
                'voltages': array of applied voltages,
                'currents': array of total currents at each voltage,
                'num_paths': array of number of active paths at each voltage,
                'conductances': array of total conductance at each voltage,
                'path_details': list of dicts with per-voltage path information
            }
        """
        if not self.source_nodes or not self.drain_nodes:
            raise ValueError("Must identify sources and drains first using identify_sources_drains()")

        print(f"\n{'='*70}")
        print("CALCULATING I-V CURVE")
        print(f"{'='*70}")
        print(f"Voltage range: {V_start}V to {V_max}V (step: {V_step}V)")
        print(f"Sources: {len(self.source_nodes)} nodes, Drains: {len(self.drain_nodes)} nodes")

        voltages = []
        currents = []
        num_paths_list = []
        conductances = []
        path_details = []

        V = V_start
        step_num = 0

        while V <= V_max:
            step_num += 1
            voltages.append(V)

            # Activate nodes based on threshold voltage
            activated_nodes = set()
            for node in self.G.nodes():
                if self.G.nodes[node]['Vth'] <= V:
                    activated_nodes.add(node)

            # Activate edges where both endpoint nodes are activated
            activated_edges = set()
            for i, j in self.G.edges():
                if i in activated_nodes and j in activated_nodes:
                    activated_edges.add((i, j))

            # Find all percolation paths
            paths = self.find_all_percolation_paths(activated_nodes, activated_edges)
            num_paths = len(paths)
            num_paths_list.append(num_paths)

            total_current = 0.0
            total_conductance = 0.0
            path_info = []

            if num_paths > 0:
                for path in paths:
                    path_R, path_I = self._calculate_path_current(path, V, tunneling_effects=tunneling_effects)
                    total_current += path_I

                    if path_R > 0:
                        path_G = 1.0 / path_R
                        total_conductance += path_G

                    # Store path details
                    path_Vth_max = max(self.G.nodes[node]['Vth'] for node in path)
                    path_info.append({
                        'path': path,
                        'resistance': path_R,
                        'current': path_I,
                        'Vth_max': path_Vth_max,
                        'length': len(path)
                    })

            currents.append(total_current)
            conductances.append(total_conductance)
            path_details.append(path_info)

            # Progress reporting
            if step_num == 1 or step_num % 20 == 0 or num_paths > 0:
                if num_paths > 0:
                    print(f"  V = {V:.3f}V: {num_paths} paths, I_total = {total_current*1e6:.3f} µA, G_total = {total_conductance*1e3:.3f} mS")
                else:
                    print(f"  V = {V:.3f}V: No percolation")

            V += V_step

        print(f"\n{'='*70}")
        print("I-V CALCULATION COMPLETE")
        print(f"{'='*70}")

        # Find percolation voltage
        percolation_idx = next((i for i, n in enumerate(num_paths_list) if n > 0), None)
        if percolation_idx is not None:
            print(f"Percolation voltage: {voltages[percolation_idx]:.3f}V")
            max_current = max(currents)
            max_idx = currents.index(max_current)
            print(f"Maximum current: {max_current*1e6:.3f} µA at V = {voltages[max_idx]:.3f}V")
        else:
            print("No percolation found in voltage range")

        return {
            'voltages': np.array(voltages),
            'currents': np.array(currents),
            'num_paths': np.array(num_paths_list),
            'conductances': np.array(conductances),
            'path_details': path_details
        }

    def _solve_kirchhoff(self, activated_nodes, V_applied):
        """
        Solve Kirchhoff nodal analysis for one voltage step.

        Builds a conductance matrix from the activated subgraph, applies
        Dirichlet boundary conditions (source nodes at V_applied, drain nodes
        at 0V), and solves for node potentials. Total current is then computed
        as the sum of currents flowing into all drain nodes.

        Parameters
        ----------
        activated_nodes : set
            Node IDs where Vth <= V_applied (the conducting subgraph)
        V_applied : float
            Applied voltage held at source nodes

        Returns
        -------
        tuple : (total_current: float, node_potentials: dict)
            total_current   -- current into drain nodes (Amperes)
            node_potentials -- {node_id: potential} for all working nodes
        """
        from scipy.sparse import lil_matrix
        from scipy.sparse.linalg import spsolve
        import networkx as nx

        if not activated_nodes:
            return 0.0, {}

        active_sources = [n for n in self.source_nodes if n in activated_nodes]
        active_drains  = [n for n in self.drain_nodes  if n in activated_nodes]
        if not active_sources or not active_drains:
            return 0.0, {}

        # Subgraph restricted to activated nodes (edges only where both endpoints active)
        G_sub = self.G.subgraph(activated_nodes)

        # Find nodes connected to BOTH a source and a drain component
        reachable_from_source = set()
        for src in active_sources:
            if src in G_sub:
                reachable_from_source |= nx.node_connected_component(G_sub, src)

        reachable_from_drain = set()
        for drn in active_drains:
            if drn in G_sub:
                reachable_from_drain |= nx.node_connected_component(G_sub, drn)

        # Only nodes bridging source to drain are meaningful
        working_nodes = reachable_from_source & reachable_from_drain
        if not working_nodes:
            return 0.0, {}

        working_list = sorted(working_nodes)
        N = len(working_list)
        local_idx = {node: k for k, node in enumerate(working_list)}

        R_MIN = 1.0  # Ω floor to avoid singularity for zero-resistance edges

        # Detect whether node-resistance model is active
        has_node_resistance = any(
            self.G.nodes[n].get('R_node', 0.0) > 0.0
            for n in working_nodes
        )

        if not has_node_resistance:
            # Edge-only model: standard N×N nodal admittance matrix
            G_mat = lil_matrix((N, N), dtype=float)
            for (i, j) in G_sub.edges():
                if i not in local_idx or j not in local_idx:
                    continue
                R = max(self.G[i][j]['R_edge'], R_MIN)
                g = 1.0 / R
                ki, kj = local_idx[i], local_idx[j]
                G_mat[ki, ki] += g;  G_mat[kj, kj] += g
                G_mat[ki, kj] -= g;  G_mat[kj, ki] -= g

            b = np.zeros(N)
            for src in active_sources:
                if src in local_idx:
                    k = local_idx[src]
                    G_mat[k, :] = 0;  G_mat[k, k] = 1.0;  b[k] = V_applied
            for drn in active_drains:
                if drn in local_idx:
                    k = local_idx[drn]
                    G_mat[k, :] = 0;  G_mat[k, k] = 1.0;  b[k] = 0.0

            try:
                phi = spsolve(G_mat.tocsr(), b)
                if not np.all(np.isfinite(phi)):
                    return 0.0, {}
            except Exception:
                return 0.0, {}

            total_current = 0.0
            for drn in active_drains:
                if drn not in local_idx:
                    continue
                k_drn = local_idx[drn]
                for nbr in G_sub.neighbors(drn):
                    if nbr not in local_idx:
                        continue
                    g = 1.0 / max(self.G[drn][nbr]['R_edge'], R_MIN)
                    total_current += g * (phi[local_idx[nbr]] - phi[k_drn])

            node_potentials = {working_list[k]: phi[k] for k in range(N)}
            return total_current, node_potentials

        else:
            # Node-resistance model: node-splitting — each internal node i
            # becomes i_in (row 2k) and i_out (row 2k+1) connected by R_node.
            # Electrodes are NOT split: they get a single row at offset 2*N_int.
            #
            # Edge orientation: BFS from active sources determines which end of
            # each undirected edge is "source-side".  We stamp ONE symmetric
            # conductance element source_out ↔ drain_in per edge so that
            # current enters each node at i_in (before R_node) and exits at
            # i_out (after R_node), regardless of graph-node-label order.
            all_electrodes = set(active_sources) | set(active_drains)
            internal_nodes = [n for n in working_list if n not in all_electrodes]
            electrode_nodes = [n for n in working_list if n in all_electrodes]

            N_int = len(internal_nodes)
            N_el  = len(electrode_nodes)
            MAT_SIZE = 2 * N_int + N_el

            # Row index helpers
            int_idx  = {n: i for i, n in enumerate(internal_nodes)}
            el_idx   = {n: i for i, n in enumerate(electrode_nodes)}

            def row_in(n):
                if n in int_idx:  return 2 * int_idx[n]
                return 2 * N_int + el_idx[n]

            def row_out(n):
                if n in int_idx:  return 2 * int_idx[n] + 1
                return 2 * N_int + el_idx[n]   # electrodes: in == out

            # BFS from active sources over the working subgraph to assign
            # a "depth" to every working node; used to orient edges.
            bfs_depth = {n: 999 for n in working_nodes}
            bfs_queue = list(active_sources)
            for s in active_sources:
                bfs_depth[s] = 0
            head = 0
            while head < len(bfs_queue):
                cur = bfs_queue[head]; head += 1
                for nbr in G_sub.neighbors(cur):
                    if nbr in bfs_depth and bfs_depth[nbr] == 999:
                        bfs_depth[nbr] = bfs_depth[cur] + 1
                        bfs_queue.append(nbr)

            G_mat = lil_matrix((MAT_SIZE, MAT_SIZE), dtype=float)

            # Internal node resistors (i_in ↔ i_out)
            for n in internal_nodes:
                R_n = max(self.G.nodes[n].get('R_node', 0.0), R_MIN)
                g_n = 1.0 / R_n
                ri, ro = row_in(n), row_out(n)
                G_mat[ri, ri] += g_n;  G_mat[ro, ro] += g_n
                G_mat[ri, ro] -= g_n;  G_mat[ro, ri] -= g_n

            # Edges: ONE symmetric conductance stamp per undirected edge.
            # Orient so that the source-side node's _out connects to the
            # drain-side node's _in (BFS depth breaks ties arbitrarily).
            for (i, j) in G_sub.edges():
                if i not in local_idx or j not in local_idx:
                    continue
                R_e = max(self.G[i][j]['R_edge'], R_MIN)
                g_e = 1.0 / R_e
                # source-side = shallower BFS depth
                if bfs_depth[i] <= bfs_depth[j]:
                    src_n, drn_n = i, j
                else:
                    src_n, drn_n = j, i
                r_src_out = row_out(src_n)
                r_drn_in  = row_in(drn_n)
                G_mat[r_src_out, r_src_out] += g_e
                G_mat[r_drn_in,  r_drn_in]  += g_e
                G_mat[r_src_out, r_drn_in]  -= g_e
                G_mat[r_drn_in,  r_src_out] -= g_e

            # Boundary conditions: pin both in and out rows of electrodes
            b = np.zeros(MAT_SIZE)
            for src in active_sources:
                if src not in el_idx:
                    continue
                for row in (row_in(src), row_out(src)):
                    G_mat[row, :] = 0;  G_mat[row, row] = 1.0;  b[row] = V_applied
            for drn in active_drains:
                if drn not in el_idx:
                    continue
                for row in (row_in(drn), row_out(drn)):
                    G_mat[row, :] = 0;  G_mat[row, row] = 1.0;  b[row] = 0.0

            try:
                phi = spsolve(G_mat.tocsr(), b)
                if not np.all(np.isfinite(phi)):
                    return 0.0, {}
            except Exception:
                return 0.0, {}

            # Total current: sum currents flowing into each drain node.
            # Each drain is an electrode (not split); its neighbors connect
            # via edge conductances from neighbor_out to drain_row.
            total_current = 0.0
            for drn in active_drains:
                if drn not in el_idx:
                    continue
                drn_row = row_in(drn)   # == row_out(drn) for electrode
                for nbr in G_sub.neighbors(drn):
                    if nbr not in local_idx:
                        continue
                    R_e = max(self.G[drn][nbr]['R_edge'], R_MIN)
                    g_e = 1.0 / R_e
                    # nbr is source-side (closer to source), drn is drain-side
                    total_current += g_e * (phi[row_out(nbr)] - phi[drn_row])

            # Return node potential as average of i_in and i_out for internal nodes
            node_potentials = {}
            for n in internal_nodes:
                node_potentials[n] = 0.5 * (phi[row_in(n)] + phi[row_out(n)])
            for n in electrode_nodes:
                node_potentials[n] = phi[row_in(n)]
            return total_current, node_potentials

    def calculate_iv_curve_kirchhoff(self, V_start=0.0, V_max=2.0, V_step=0.01):
        """
        Calculate I-V curve using Kirchhoff nodal analysis (conductance matrix).

        Unlike the BFS percolation method, this approach:
        - Allows current to flow through shared nodes (no node-disjoint restriction)
        - Treats Vth as a pure gating threshold (not a voltage barrier subtraction)
        - Solves the full resistor network exactly via nodal analysis

        Parameters
        ----------
        V_start : float
            Starting voltage
        V_max : float
            Maximum voltage
        V_step : float
            Voltage increment

        Returns
        -------
        dict with keys: 'voltages', 'currents', 'conductances',
                        'num_paths' (zeros, N/A), 'path_details' (empty, N/A)
        """
        if not self.source_nodes or not self.drain_nodes:
            raise ValueError("Must identify sources and drains first using identify_sources_drains()")

        print(f"\n{'='*70}")
        print("I-V CALCULATION (KIRCHHOFF NODAL ANALYSIS)")
        print(f"{'='*70}")

        voltages, currents, conductances = [], [], []

        V = V_start
        step_num = 0
        while V <= V_max + 1e-10:
            step_num += 1

            activated_nodes = {n for n in self.G.nodes() if self.G.nodes[n]['Vth'] <= V}
            total_current, _ = self._solve_kirchhoff(activated_nodes, V)

            voltages.append(V)
            currents.append(total_current)
            conductances.append(total_current / V if V > 1e-12 else 0.0)

            if step_num == 1 or step_num % 20 == 0 or total_current > 0:
                if total_current > 0:
                    print(f"  V = {V:.3f}V: I = {total_current*1e9:.3f} nA, "
                          f"G = {(total_current/V)*1e12:.3f} pS, "
                          f"{len(activated_nodes)} nodes active")
                else:
                    print(f"  V = {V:.3f}V: No conduction, {len(activated_nodes)} nodes active")

            V = round(V + V_step, 10)

        print(f"\n{'='*70}")
        print("KIRCHHOFF CALCULATION COMPLETE")
        print(f"{'='*70}")

        percolation_idx = next((i for i, c in enumerate(currents) if c > 0), None)
        if percolation_idx is not None:
            print(f"Percolation voltage: {voltages[percolation_idx]:.3f}V")
            print(f"Maximum current: {max(currents)*1e9:.3f} nA at V = {voltages[currents.index(max(currents))]:.3f}V")
        else:
            print("No percolation found in voltage range")

        return {
            'voltages':     np.array(voltages),
            'currents':     np.array(currents),
            'conductances': np.array(conductances),
            'num_paths':    np.zeros(len(voltages)),
            'path_details': [[] for _ in voltages],
        }
