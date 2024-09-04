import cvxopt
import mosek


class LPSolver:

    def __init__(self, epsilon=0.2):
        self.c = []
        # constraints
        self.lhs_triples = []
        self.rhs = []
        self.epsilon = epsilon

    def add_deployment(self, throughput_per_replica, input_rate, target_latency):
        # utility
        var_idx = len(self.c)
        coeff = throughput_per_replica * target_latency / input_rate
        # print(coeff)
        # (value, row_index, column_index)
        # x_i >= 1 => easily infeasible due to low input rate deployment
        # x_i >= 0 and then ceil(x_i) for the deployment < 1.0
        self.lhs_triples.append((-1., len(self.lhs_triples), var_idx))
        self.rhs.append(0.)
        # max utility
        self.lhs_triples.append((coeff, len(self.lhs_triples), var_idx))
        self.rhs.append(1. + self.epsilon)
        self.c.append(-coeff)  # maximize

    def run(self, max_num_replicas=None):
        c = cvxopt.matrix(self.c)

        if max_num_replicas is not None:
            # TODO
            raise NotImplementedError()

        lhs = cvxopt.spmatrix(*zip(*self.lhs_triples))
        rhs = cvxopt.matrix(self.rhs)

        cvxopt.solvers.options['mosek'] = {
            mosek.dparam.optimizer_max_time:  100.0,
            mosek.iparam.intpnt_solve_form:   mosek.solveform.dual}
        cvxopt.solvers.options['verbose'] = False
        solution = cvxopt.solvers.lp(c, lhs, rhs, solver="mosek")

        print(solution["x"])


if __name__ == "__main__":
    throughput_per_replica = 5  # 0.2 per requests
    target_latency = 0.8
    input_rates = [1, 5, 2, 10, 4]
    solver = LPSolver()
    for input_rate in input_rates:
        solver.add_deployment(throughput_per_replica,
                              input_rate, target_latency)
    solver.run()
