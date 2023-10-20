from dataclasses import dataclass

@dataclass
class TabuParams:
    tabu_list_size: int = 10
    max_iterations: int = 100

@dataclass
class CandidateSolution:
    pass

def tabu_serach(jobs: list[], params: TabuParams = TabuParams()) -> :

    initial_solution = _get_intial_solution()
    best_solution = _tabu_search(initial_solution, params)
    return best_solution



def _tabu_search(initial_solution: CandidateSolution, params:TabuParams = TabuParams()) -> CandidateSolution:
    current_solution = initial_solution
    best_solution = initial_solution
    tabu_list = []
    iteration = 0

    while iteration < params.max_iterations:
        candidate_solutions = _generate_candidate_solutions(current_solution)
        non_tabu_solutions = [s for s in candidate_solutions if s not in tabu_list]
        if len(non_tabu_solutions) == 0:
            return best_solution
        else:
            next_solution = _select_best_solution(non_tabu_solutions)
            if _evaluate_solution(next_solution) > _evaluate_solution(best_solution):
                best_solution = next_solution
            tabu_list.append(next_solution)
            if len(tabu_list) > params.tabu_list_size:
                tabu_list.pop(0)
            current_solution = next_solution
            iteration += 1

    return best_solution

def _generate_candidate_solutions(solution):
    # generate a list of candidate solutions based on the current solution
    pass

def _select_best_solution(solutions):
    # select the best solution from a list of candidate solutions
    pass

def _evaluate_solution(solution):
    # evaluate the quality of a solution
    pass

def _get_intial_solution():
    # initialize a solution using shortest processing time
    pass

def _convert_solution_to_jobs(solution):
    # convert a solution to a list of jobs
    pass
