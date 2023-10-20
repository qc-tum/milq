import heapq

def branch_and_bound(graph, start, goal):
    # Initialize the priority queue with the starting node
    queue = [(0, [start])]
    # Initialize the best path and its cost
    best_path = None
    best_cost = float('inf')
    
    while queue:
        # Get the next node to explore
        cost, path = heapq.heappop(queue)
        node = path[-1]
        
        # If we've reached the goal, update the best path and its cost
        if node == goal:
            if cost < best_cost:
                best_path = path
                best_cost = cost
        else:
            # Otherwise, explore the node's neighbors
            for neighbor, neighbor_cost in graph[node].items():
                # Only explore the neighbor if it hasn't been visited before
                if neighbor not in path:
                    # Compute the new cost and add the neighbor to the path
                    new_cost = cost + neighbor_cost
                    new_path = path + [neighbor]
                    # Add the new path to the priority queue
                    heapq.heappush(queue, (new_cost, new_path))
    
    return best_path, best_cost
