import math
import random
from collections import defaultdict, deque
from heapq import heappush, heappop
import numpy as np
import networkx as nx
from tqdm import tqdm


class VehicularEdgeComputingSystem:
    
    def __init__(self, num_vehicles, num_mec_servers, num_subtasks, 
                 area_size=1500, v2v_range=150, v2m_range=200, max_hops=4, seed=0):
        """
        Initialize the system parameters.
        
        Args:
            num_vehicles (int): Number of vehicles |V|
            num_mec_servers (int): Number of MEC servers |M|
            num_subtasks (int): Number of subtasks |N|
            area_size (float): Size of the side length (meters)
            v2v_range (float): V2V communication range (meters)
            v2m_range (float): V2MEC communication range (meters)
            max_hops (int): Maximum number of hops allowed (default: 4)
            seed (int): Reproducibility
        """
        random.seed(seed)
        np.random.seed(seed)

        self.V = num_vehicles
        self.M = num_mec_servers
        self.C = num_vehicles + num_mec_servers
        self.n = num_subtasks
        self.max_hops = max_hops
        
        self.area_size = area_size
        self.v2v_range = v2v_range
        self.v2m_range = v2m_range
        
        self.positions = self._generate_positions()
        
        self.F_vehicle = 2e9 
        self.F_mec = 10e9
        
        self.xi_V_dBm = 23
        self.xi_VM_dBm = 46
        
        self.N0_dBm_Hz = -174
        
        self.B_v2v = 25e6
        self.B_v2m = 30e6
        self.B_m2m = 1e9
        
        
        self.W = [random.uniform(1.6e6, 10e6) for _ in range(num_subtasks)]
        
        self.task_graph = self._generate_tree_task_graph()
        self.sink_nodes = self._find_sink_nodes()
        
        self.physical_graph = self._generate_physical_graph()
        self.communication_graph, self.equivalent_rates = self._compute_multihop_graph()
        
        self.data_transfer_size = [[0] * num_subtasks for _ in range(num_subtasks)]
        self._initialize_data_transfer_sizes()
        
        self.total_communication_rate = self._compute_total_communication_rate()
        
        self.assignment = [-1] * num_subtasks
        self.best_assignment = [-1] * num_subtasks
        
    def _generate_positions(self):
        """Generate random positions for vehicles and grid-based for MEC servers."""
        positions = []
        
        # Vehicles
        for _ in range(self.V):
            x = random.uniform(0, self.area_size)
            y = random.uniform(0, self.area_size)
            positions.append((x, y))
        
        # MEC servers
        if self.M > 0:
            grid_size = int(math.ceil(math.sqrt(self.M)))
            step = self.area_size / (grid_size + 1)
            
            mec_count = 0
            for i in range(grid_size):
                for j in range(grid_size):
                    if mec_count >= self.M:
                        break
                    x = (i + 1) * step
                    y = (j + 1) * step
                    positions.append((x, y))
                    mec_count += 1
        
        return positions
    
    def _generate_physical_graph(self):
        """
        Generate physical communication graph (direct single-hop connections only).
        Returns adjacency list with edge rates.
        """
        adj = defaultdict(dict)
        
        for i in range(self.C):
            adj[i][i] = float('inf')
        
        for i in range(self.C):
            for j in range(i + 1, self.C):
                distance = self._compute_distance(i, j)
                connected = False
                rate = 0
                
                # V2V
                if i < self.V and j < self.V:
                    if distance <= self.v2v_range:
                        connected = True
                        rate = self._compute_single_hop_rate(i, j)
                
                # V2MEC
                elif (i < self.V and j >= self.V) or (i >= self.V and j < self.V):
                    if distance <= self.v2m_range:
                        connected = True
                        rate = self._compute_single_hop_rate(i, j)
                
                # M2M
                else:
                    if j == i + 1:
                        connected = True
                        rate = self.B_m2m
                
                if connected:
                    adj[i][j] = rate
                    adj[j][i] = rate
        
        return adj

    def _compute_distance(self, device_i, device_j):
        """Compute Euclidean distance between two devices."""
        x1, y1 = self.positions[device_i]
        x2, y2 = self.positions[device_j]
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def _compute_single_hop_rate(self, device_i, device_j):
        """
        Compute single-hop communication rate using Shannon capacity.
        """
        if device_i == device_j:
            return float('inf')
        
        # V2V
        if device_i < self.V and device_j < self.V:
            bandwidth = self.B_v2v
            P_tx = self.xi_V_dBm
            N0 = self.N0_dBm_Hz
            path_loss_dB = self._compute_path_loss_dB(device_i, device_j)
            SNR = 10**((P_tx - path_loss_dB - (N0 + 10*math.log10(bandwidth))) / 10)
            
            rate = bandwidth * math.log2(1 + SNR)
            return rate 
        
        # V2MEC
        elif (device_i < self.V and device_j >= self.V) or (device_i >= self.V and device_j < self.V):
            bandwidth = self.B_v2m
            P_tx = self.xi_VM_dBm
            N0 = self.N0_dBm_Hz
            path_loss_dB = self._compute_path_loss_dB(device_i, device_j)
            SNR = 10**((P_tx - path_loss_dB - (N0 + 10*math.log10(bandwidth))) / 10)
            
            rate = bandwidth * math.log2(1 + SNR)
            return rate
        
        # M2M
        else:
            return self.B_m2m
        
    def _compute_path_loss_dB(self, device_i, device_j):
        """
        Compute path loss in dB using the provided formulas:
        - V2V: P_loss = 63.3 + 17.7 * log10(D)
        - V2MEC: P_loss = 128.1 + 37.5 * log10(D)
        
        Returns:
            Path loss in dB
        """
        distance = self._compute_distance(device_i, device_j)
        
        # V2V
        if device_i < self.V and device_j < self.V:
            path_loss_dB = 63.3 + 17.7 * math.log10(distance / 1000)
        # V2MEC
        elif (device_i < self.V and device_j >= self.V) or (device_i >= self.V and device_j < self.V):
            path_loss_dB = 128.1 + 37.5 * math.log10(distance / 1000)
        else:  # M2M
            path_loss_dB = 0
        
        return path_loss_dB
    
    def _compute_multihop_graph(self):
        """
        Compute multi-hop communication graph with hop constraint H.
        
        For each pair of nodes (i, j):
        1. Find shortest path with <= H hops
        2. Compute equivalent rate: 1/R_equiv = sum(1/R_k) for each edge k in path
        3. Store in compressed single-hop graph
        
        Returns:
            (communication_graph, equivalent_rates)
            - communication_graph: adjacency list
            - equivalent_rates: dict mapping (i,j) -> equivalent rate
        """
        print(f"Computing multi-hop paths with max_hops={self.max_hops}...")
        
        G = nx.Graph()
        for i in range(self.C):
            G.add_node(i)
        for u in self.physical_graph:
            for v, rate in self.physical_graph[u].items():
                if u != v and rate > 0:
                    G.add_edge(u, v, weight=1.0/rate, rate=rate)
        
        communication_graph = defaultdict(list)
        equivalent_rates = {}
        
        for i in range(self.C):
            # Self-loop
            communication_graph[i].append(i)
            equivalent_rates[(i, i)] = float('inf')
            
            for j in range(i + 1, self.C):
                if j in self.physical_graph[i]:
                    rate = self.physical_graph[i][j]
                    communication_graph[i].append(j)
                    communication_graph[j].append(i)
                    equivalent_rates[(i, j)] = rate
                    equivalent_rates[(j, i)] = rate
                else:
                    try:
                        path = nx.shortest_path(G, i, j, weight='weight')
                        num_hops = len(path) - 1
                        if num_hops <= self.max_hops:
                            inverse_rate_sum = 0
                            for k in range(len(path) - 1):
                                u, v = path[k], path[k + 1]
                                edge_rate = self.physical_graph[u][v]
                                inverse_rate_sum += 1.0 / edge_rate
                            equiv_rate = 1.0 / inverse_rate_sum
                            
                            communication_graph[i].append(j)
                            communication_graph[j].append(i)
                            equivalent_rates[(i, j)] = equiv_rate
                            equivalent_rates[(j, i)] = equiv_rate
                    
                    except nx.NetworkXNoPath:
                        pass
        
        num_connections = sum(len(neighbors) - 1 for neighbors in communication_graph.values()) // 2
        print(f"Multi-hop graph computed: {num_connections} connections (including multi-hop)")
        
        return communication_graph, equivalent_rates
    
    def _generate_tree_task_graph(self):
        """Generate tree-structured task dependency graph."""
        if self.n <= 1:
            return defaultdict(list)
        
        adj_list = defaultdict(list)
        nodes = list(range(self.n))
        
        for i in range(1, self.n):
            child = nodes[i]
            parent = random.choice(nodes[:i])
            adj_list[child].append(parent)
            
        return adj_list
    
    def _find_sink_nodes(self):
        """Find sink nodes (leaf nodes in the tree)."""
        has_parent = set()
        for child in self.task_graph:
            if len(self.task_graph[child]) > 0:
                has_parent.add(child)
        return set(range(self.n)) - has_parent
    
    def _initialize_data_transfer_sizes(self):
        """Initialize data transfer sizes"""
        for child in self.task_graph:
            for parent in self.task_graph[child]:
                self.data_transfer_size[child][parent] = random.uniform(8e6, 50e6)
    
    def _compute_total_communication_rate(self):
        """Compute sum of all communication rates for priority calculation."""
        total = 0
        for (i, j), rate in self.equivalent_rates.items():
            if i < j:
                total += rate
        return total if total > 0 else 1.0
    
    def _compute_communication_rate(self, device_i, device_j):
        """
        Get communication rate from precomputed equivalent rates.
        """
        if device_i == device_j:
            return float('inf')
        
        if (device_i, device_j) in self.equivalent_rates:
            return self.equivalent_rates[(device_i, device_j)]
        else:
            return 0
    
    def _compute_computation_time(self, subtask, device):
        """Compute computation time: Δ = W / F."""
        if device < self.V:
            return self.W[subtask] / self.F_vehicle
        else:
            return self.W[subtask] / self.F_mec
    
    def _compute_transmission_time(self, subtask_i, subtask_j, device_i, device_j):
        """Compute transmission time: T = S / R."""
        if device_i == device_j:
            return 0
        
        data_size = self.data_transfer_size[subtask_i][subtask_j]
        rate = self._compute_communication_rate(device_i, device_j)
        
        if rate == 0:
            return float('inf')
        
        return data_size / rate
    
    def _get_predecessors(self, graph):
        """Get predecessor nodes."""
        pred = defaultdict(set)
        for child in graph:
            for parent in graph[child]:
                pred[parent].add(child)
        return pred
    
    def _get_successors(self, graph):
        """Get successor nodes."""
        succ = defaultdict(set)
        for child in graph:
            for parent in graph[child]:
                succ[child].add(parent)
        return succ
    
    def _get_undirected_neighbors(self, graph):
        """Convert directed graph to undirected."""
        undirected = defaultdict(set)
        for child in graph:
            for parent in graph[child]:
                undirected[child].add(parent)
                undirected[parent].add(child)
        return undirected
    
    # ========================================================================
    # Algorithm 1: Execution Priority
    # ========================================================================
    
    def compute_execution_priority(self):
        """
        Algorithm 1: Compute execution priority (bottom-up from sink).
        
        P(n) = max_{n' in Succ(n)} {P(n') + avg_comp_n + avg_comm_{n,n'}}
        """
        reversed_graph = defaultdict(list)
        for child in self.task_graph:
            for parent in self.task_graph[child]:
                reversed_graph[parent].append(child)
        
        pred = self._get_predecessors(reversed_graph)
        succ = self._get_successors(reversed_graph)
        
        P = [0.0] * self.n
        pending = set(self.sink_nodes)
        visited = set()
        
        while pending:
            current_pending = list(pending)
            
            for n in current_pending:
                if all(pred_n in visited for pred_n in pred[n]):
                    pending.remove(n)
                    
                    max_priority = 0
                    for n_pred in pred[n]:
                        # Average computation time
                        avg_comp = sum(self._compute_computation_time(n, c) 
                                      for c in range(self.C)) / self.C
                        
                        # Average communication time
                        total_comm_time = 0
                        num_pairs = 0
                        for c1 in range(self.C):
                            for c2 in range(self.C):
                                if c1 != c2 and c2 in self.communication_graph[c1]:
                                    rate = self._compute_communication_rate(c1, c2)
                                    if rate > 0:
                                        total_comm_time += self.data_transfer_size[n_pred][n] / rate
                                        num_pairs += 1
                        
                        avg_comm = total_comm_time / num_pairs if num_pairs > 0 else 0
                        
                        priority = P[n_pred] + avg_comp + avg_comm
                        max_priority = max(max_priority, priority)
                    
                    if max_priority == 0:
                        max_priority = sum(self._compute_computation_time(n, c) 
                                         for c in range(self.C)) / self.C
                    
                    P[n] = max_priority
                    visited.add(n)
                    
                    for n_succ in succ[n]:
                        if n_succ not in visited and n_succ not in pending:
                            pending.add(n_succ)
        return P
    
    # ========================================================================
    # Algorithm 3: Reassign
    # ========================================================================
    
    def reassign(self, visited_priority_queue, current_subtask, visited_set, verbose=False):
        """
        Algorithm 3: Recompute execution times based on current assignment.
        """
        AFT = [0.0] * self.n
        EST = [0.0] * self.n
        arrive_time = [0.0] * self.n
        t_remain = [0.0] * self.C
        compute_time = [0.0] * self.n
        
        pred = self._get_predecessors(self.task_graph)
        
        import heapq
        
        while visited_priority_queue:
            _, n = heapq.heappop(visited_priority_queue)
            if n == current_subtask:
                c = self.assignment[n]
            else:
                c = self.best_assignment[n]
            
            if c == -1:
                continue

            EST[n] = 0
            for pred_n in pred[n]:
                if pred_n in visited_set:
                    pred_device = (self.assignment[pred_n] if pred_n == current_subtask 
                                 else self.best_assignment[pred_n])
                    if pred_device != -1:
                        trans_time = self._compute_transmission_time(pred_n, n, pred_device, c)
                        EST[n] = max(EST[n], AFT[pred_n] + trans_time)
            
            arrive_time[n] = EST[n]
            EST[n] = max(EST[n], t_remain[c])
            
            comp_time = self._compute_computation_time(n, c)
            AFT[n] = EST[n] + comp_time
            compute_time[n] = comp_time
            
            t_remain[c] = AFT[n]
        
        completion_time = max(AFT) if AFT else 0
        avg_process_time = sum(AFT[i] - arrive_time[i] for i in range(self.n)) / self.n if self.n > 0 else 0
        
        utilization = [0.0] * self.C
        for i in range(self.n):
            if self.best_assignment[i] != -1:
                utilization[self.best_assignment[i]] += compute_time[i]
        avg_utilization = sum(utilization) / self.C / completion_time * 100 if completion_time > 0 else 0
        
        if verbose:
            print(f"Completion time: {completion_time:.6f}s")
        
        return completion_time, avg_process_time, avg_utilization
    
    # ========================================================================
    # Algorithm 2: BFS-based Subtask Offloading
    # ========================================================================
    
    def bfs_based_offloading(self, priority_values=None, ego_vehicle=0):
        """
        Algorithm 2: BFS-based subtask offloading with greedy device selection.
        """
        if priority_values is None:
            priority_values = self.compute_execution_priority()
        
        priority_array = np.array(priority_values)
        
        n_star = priority_array.argmax()
        priority_star = priority_array.max()
        
        pq = []
        heappush(pq, (-priority_star, n_star, None))
        
        visited_pq = []
        
        self.assignment = [-1] * self.n
        self.best_assignment = [-1] * self.n
        
        visited = set()
        in_pq = set([n_star])
        
        undirected_graph = self._get_undirected_neighbors(self.task_graph)
        
        while pq:
            priority, current, source = heappop(pq)
            
            visited.add(current)
            heappush(visited_pq, (priority, current))
            
            if source is None:
                candidates = list(self.communication_graph[ego_vehicle])
            else:
                candidates = list(self.communication_graph[self.best_assignment[source]])
            
            min_completion_time = float('inf')
            best_device = None
            
            for c in candidates:
                self.assignment[current] = c
                
                completion_time, _, _ = self.reassign(
                    visited_pq.copy(), current, visited
                )
                
                if completion_time < min_completion_time:
                    min_completion_time = completion_time
                    best_device = c
            
            if best_device is not None:
                self.best_assignment[current] = best_device
            else:
                self.best_assignment[current] = 0
            
            for neighbor in undirected_graph[current]:
                if neighbor not in in_pq:
                    heappush(pq, (-priority_array[neighbor], neighbor, current))
                    in_pq.add(neighbor)
        
        final_completion, avg_process, avg_util = self.reassign(
            visited_pq.copy(), -1, visited
        )
        
        return final_completion, avg_process, avg_util
    
    # ========================================================================
    # Baseline Algorithms
    # ========================================================================
    
    def random_offloading(self, ego_vehicle=0):
        """Random device selection baseline."""
        priority_values = self.compute_execution_priority()
        priority_array = np.array(priority_values)
        
        n_star = priority_array.argmax()
        priority_star = priority_array.max()
        
        pq = []
        heappush(pq, (-priority_star, n_star, None))
        visited_pq = []
        
        self.assignment = [-1] * self.n
        self.best_assignment = [-1] * self.n
        
        visited = set()
        in_pq = set([n_star])
        undirected_graph = self._get_undirected_neighbors(self.task_graph)
        
        while pq:
            priority, current, source = heappop(pq)
            visited.add(current)
            heappush(visited_pq, (priority, current))
            
            if source is None:
                candidates = list(self.communication_graph[ego_vehicle])
            else:
                candidates = list(self.communication_graph[self.best_assignment[source]])
            
            if candidates:
                selected_device = random.choice(candidates)
                self.best_assignment[current] = selected_device
                self.assignment[current] = selected_device
            
            for neighbor in undirected_graph[current]:
                if neighbor not in in_pq:
                    heappush(pq, (-priority_array[neighbor], neighbor, current))
                    in_pq.add(neighbor)
        
        return self.reassign(visited_pq.copy(), -1, visited)
    

    def find_max_clique_networkx(self, G, required_nodes):
        all_cliques = list(nx.find_cliques(G))
        required_set = set([required_nodes])
        valid_cliques = [
            clique for clique in all_cliques 
            if required_set.issubset(set(clique))
        ]
    
        if valid_cliques:
            return max(valid_cliques, key=len)
        else:
            return required_set


    def pbtsa_baseline(self, ego_vehicle=0):
        """PBTSA with maximum clique."""
        G = nx.Graph()
        edges = []
        visited_edges = set()
        
        for u in self.communication_graph:
            for v in self.communication_graph[u]:
                if (u, v) not in visited_edges and (v, u) not in visited_edges:
                    visited_edges.add((u, v))
                    edges.append((u, v))
        
        G.add_edges_from(edges)
        max_clique = self.find_max_clique_networkx(G, required_nodes=ego_vehicle)

        priority_values = self.compute_execution_priority()
        order = [(pr, idx) for idx, pr in enumerate(priority_values)]
        order.sort(reverse=True)
        
        AFT = [0.0] * self.n
        EST = [0.0] * self.n
        t_remain = [0.0] * self.C
        n_to_device = {}
        
        pred = self._get_predecessors(self.task_graph)
        
        for _, ni in order:
            AFT[ni] = float('inf')
            best_device = None
            
            for c in max_clique:
                EST[ni] = 0
                for nj in pred[ni]:
                    if nj in n_to_device:
                        trans_time = self._compute_transmission_time(nj, ni, n_to_device[nj], c)
                        EST[ni] = max(EST[ni], AFT[nj] + trans_time)
                    else:
                        raise ValueError(f"Predecessor {nj} of {ni} has not been assigned a device yet.")
                
                EST[ni] = max(EST[ni], t_remain[c])
                EFT = EST[ni] + self._compute_computation_time(ni, c)
                
                if EFT < AFT[ni]:
                    AFT[ni] = EFT
                    best_device = c
            
            if best_device is not None:
                t_remain[best_device] = AFT[ni]
                n_to_device[ni] = best_device
        
        completion_time = max(AFT) if AFT else 0
        avg_process = sum(AFT) / self.n if self.n > 0 else 0
        
        utilization = [0.0] * self.C
        for i in range(self.n):
            if i in n_to_device:
                utilization[n_to_device[i]] += self._compute_computation_time(i, n_to_device[i])
        avg_util = sum(utilization) / self.C / completion_time * 100 if completion_time > 0 else 0
        return completion_time, avg_process, avg_util


# ============================================================================
# Experiment Functions
# ============================================================================

def run_experiment(num_vehicles, num_mec_servers, num_subtasks, 
                   num_trials=10, max_hops=4 ,**kwargs):
    """Run experiment with multiple trials and return averaged results."""
    results = {
        'ours': {'latency': [], 'process_time': [], 'utilization': []},
        'random': {'latency': [], 'process_time': [], 'utilization': []},
        'pbtsa': {'latency': [], 'process_time': [], 'utilization': []}
    }
    
    for seed in range(num_trials):
        
        system = VehicularEdgeComputingSystem(
            num_vehicles=num_vehicles,
            num_mec_servers=num_mec_servers,
            num_subtasks=num_subtasks,
            seed=seed,
            max_hops=max_hops,
            **kwargs
        )
        ego_vehicle = random.randint(0, num_vehicles - 1)
        try:
            latency, process_time, util = system.bfs_based_offloading(ego_vehicle=ego_vehicle)
            results['ours']['latency'].append(latency)
            results['ours']['process_time'].append(process_time)
            results['ours']['utilization'].append(util)
        except Exception as e:
            print(f"Error in ours (seed {seed}): {e}")
        
        try:
            latency, process_time, util = system.random_offloading(ego_vehicle=ego_vehicle)
            results['random']['latency'].append(latency)
            results['random']['process_time'].append(process_time)
            results['random']['utilization'].append(util)
        except Exception as e:
            print(f"Error in random (seed {seed}): {e}")
        
        try:
            latency, process_time, util = system.pbtsa_baseline(ego_vehicle=ego_vehicle)
            results['pbtsa']['latency'].append(latency)
            results['pbtsa']['process_time'].append(process_time)
            results['pbtsa']['utilization'].append(util)
        except Exception as e:
            print(f"Error in PBTSA (seed {seed}): {e}")
    
    averaged_results = {}
    for method in results:
        averaged_results[method] = {
            'latency': np.mean(results[method]['latency']) if results[method]['latency'] else 0,
            'process_time': np.mean(results[method]['process_time']) if results[method]['process_time'] else 0,
            'utilization': np.mean(results[method]['utilization']) if results[method]['utilization'] else 0
        }
    
    return averaged_results

def experiment_max_hop():
    """Experiment: Max hop vs latency."""
    print("Running experiment: Max Hop vs Latency")
    
    max_hops_values = [1, 2, 4, 6]
    results = {'max_hops': [], 'latency': [], 'method': []}
    
    for max_hops in tqdm(max_hops_values):
        print(f"  Max Hops: {max_hops}")
        
        exp_results = run_experiment(
            num_vehicles=50,
            num_mec_servers=10,
            num_subtasks=400,
            num_trials=20,
            area_size=1500,
            v2v_range=150,
            v2m_range=200,
            max_hops=max_hops
        )
        
        for method in ['ours', 'random', 'pbtsa']:
            results['max_hops'].append(max_hops)
            results['latency'].append(exp_results[method]['latency'])
            results['method'].append(method.upper() if method != 'ours' else 'Ours')
    print(results)

def experiment_coverage_range():
    """Experiment: Coverage range vs latency."""
    print("Running experiment: Coverage Range vs Latency")
    
    coverage_ranges = [500, 1000, 1500, 2000]
    results = {'coverage_range': [], 'latency': [], 'method': []}
    
    for coverage_range in tqdm(coverage_ranges):
        print(f"  Coverage Range: {coverage_range}m")
        
        exp_results = run_experiment(
            num_vehicles=50,
            num_mec_servers=10,
            num_subtasks=400,
            num_trials=20,
            area_size=coverage_range,
            v2v_range=150,
            v2m_range=200,
            max_hops=4
        )
        
        for method in ['ours', 'random', 'pbtsa']:
            results['coverage_range'].append(coverage_range)
            results['latency'].append(exp_results[method]['latency'])
            results['method'].append(method.upper() if method != 'ours' else 'Ours')
    print(results)


def experiment_num_subtasks():
    """Experiment: Number of subtasks vs latency (Figure 4)."""
    print("Running experiment: Number of Subtasks vs Latency")
    
    subtask_counts = [100, 250, 400, 600]
    results = {'num_subtasks': [], 'latency': [], 'method': []}
    
    for num_subtasks in tqdm(subtask_counts):
        print(f"  Subtasks: {num_subtasks}")
        
        exp_results = run_experiment(
            num_vehicles=50,
            num_mec_servers=10,
            num_subtasks=num_subtasks,
            num_trials=20,
            area_size=1500,
            v2v_range=150,
            v2m_range=200,
            max_hops=4
        )
        
        for method in ['ours', 'random', 'pbtsa']:
            results['num_subtasks'].append(num_subtasks)
            results['latency'].append(exp_results[method]['latency'])
            results['method'].append(method.upper() if method != 'ours' else 'Ours')
    print(results)


def experiment_num_vehicles():
    """Experiment: Number of vehicles vs latency."""
    print("Running experiment: Number of Vehicles vs Latency")
    
    vehicle_counts = [10, 50, 75, 100]
    results = {'num_vehicles': [], 'latency': [], 'method': []}
    
    for num_vehicles in tqdm(vehicle_counts):
        print(f"  Vehicles: {num_vehicles}")
        
        exp_results = run_experiment(
            num_vehicles=num_vehicles,
            num_mec_servers=10,
            num_subtasks=400,
            num_trials=20,
            area_size=1500,
            v2v_range=150,
            v2m_range=200,
            max_hops=4
        )
        
        for method in ['ours', 'random', 'pbtsa']:
            results['num_vehicles'].append(num_vehicles)
            results['latency'].append(exp_results[method]['latency'])
            results['method'].append(method.upper() if method != 'ours' else 'Ours')
    print(results)

def experiment_num_mec_servers():
    """Experiment: Number of MEC servers vs latency."""
    print("Running experiment: Number of MEC Servers vs Latency")
    
    mec_server_counts = [1, 5, 10, 20]
    results = {'num_mec_servers': [], 'latency': [], 'method': []}
    
    for num_mec_servers in tqdm(mec_server_counts):
        print(f"  MEC Servers: {num_mec_servers}")
        
        exp_results = run_experiment(
            num_vehicles=50,
            num_mec_servers=num_mec_servers,
            num_subtasks=400,
            num_trials=20,
            area_size=1500,
            v2v_range=150,
            v2m_range=200,
            max_hops=4
        )
        
        for method in ['ours', 'random', 'pbtsa']:
            results['num_mec_servers'].append(num_mec_servers)
            results['latency'].append(exp_results[method]['latency'])
            results['method'].append(method.upper() if method != 'ours' else 'Ours')
    print(results)

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Vehicular Edge Computing Task Offloading Simulation")
    print("=" * 70)
    print()
    # Run experiments
    experiment_coverage_range()
    print("End of experiment: Coverage Range vs Latency")
    experiment_num_subtasks()
    print("End of experiment: Number of Subtasks vs Latency")
    experiment_num_vehicles()
    print("End of experiment: Number of Vehicles vs Latency")
    experiment_num_mec_servers()
    print("End of experiment: Number of MEC Servers vs Latency")
    experiment_max_hop()
    print("End of experiment: Maximum Hop vs Latency")
    
    print()
    print("All experiments completed!")