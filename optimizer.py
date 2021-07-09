from clingo.control import Control
from clingo.symbol import Number
from operator import itemgetter
import numpy as np
import os

def model_to_program(symbols):
    return "".join(str(symbol)+"." for symbol in symbols)

class HallOfFame:
    def __init__(self, size=3):
        self.members = []
        self.size = size

    def add(self, member, quality):
        if not self.members or self.best[1] < quality:
            self.on_record(member, quality)

        self.members.append((member, quality))
        self.members.sort(reverse=True, key=itemgetter(1))
        if len(self.members) > self.size:
            self.members.pop()

    @property
    def best(self):
        return self.members[0]

    # TODO - make this more polymorphic
    def on_record(self, member, quality):
        print(f"New record : {quality}")
        #joint = member[0] + member[1]
        #print(Graph2dSquare.fromModel(joint).formatPlanMathematica())
        #with open("record.lp", "w") as sav:
        #    print("".join(str(a)+"." for a in member[0]), file=sav, flush=True)

    def constraint_to_better(self):
        # The idea of this function was to constraint further search by making bounds based on existing solutions
        # it's currently buggy and causes finding suboptimal models instead
        if self.members:
            best_complexity, best_time = self.best[1]
            return f":- solution_complexity(C), C < {best_complexity}. :- solution_complexity({best_complexity}), min_solved_at(T), T < {best_time}."
        else:
            return ""

class Graph2dSquare:
    def __init__(self, matrix, plan):
        self.matrix = matrix
        self.plan_matrix = plan

    def formatMatrixMathematica(self, matrix=None):
        if matrix is None:
            matrix = self.matrix
        rows = ("{" + ",".join(map(str, row)) + "}" for row in matrix)
        return "{" + ",".join(rows) + "}"

    def formatPlanMathematica(self):
        return "ListAnimate[{" + ",".join(f"MatrixPlot[{self.formatMatrixMathematica(matrix)}]" for matrix in self.plan_matrix) +"}]"

    @classmethod
    def fromModel(self, model_symbols):
        """ Reconstructs the matrix from a clingo model """
        components = {}
        plan = {}
        shape_x = shape_y = max_time = min_solved_at = None
        labels = {}

        for symbol in model_symbols:
            if symbol.name == "belongs_to":
                x = symbol.arguments[0].arguments[0].number
                y = symbol.arguments[0].arguments[1].number
                component = symbol.arguments[1].number

                components[(x,y)] = component

            elif symbol.name == "shape":
                shape_x = symbol.arguments[0].number
                shape_y = symbol.arguments[1].number

            elif symbol.name == "max_time":
                max_time = symbol.arguments[0].number

            elif symbol.name == "min_solved_at":
                min_solved_at = symbol.arguments[0].number

            elif symbol.name == "label_belongs_to":
                label = symbol.arguments[0].number
                component = symbol.arguments[1].number
                labels[label] = component

            elif symbol.name == "at":
                timestamp = symbol.arguments[0].number
                label = symbol.arguments[1].number
                x = symbol.arguments[2].arguments[0].number
                y = symbol.arguments[2].arguments[1].number
                plan[(timestamp, x, y)] = label


        matrix = np.zeros((shape_x, shape_y), dtype=int)
        plan_matrix = np.zeros((min_solved_at+1, shape_x*3, shape_y*3), dtype=int)
        for point, component in components.items():
            matrix[point] = component
        for (t,x,y), label in plan.items():
            if t <= min_solved_at:
                plan_matrix[(t, x+shape_x, y+shape_y)] = labels[label]

        return self(matrix, plan_matrix)

class Planner:
    def __init__(self):
        # TODO: Investigate when exactly does clingo free the model memory to avoid SIGSEGV (has happened in the past, maybe already irrelevant)
        self.models = HallOfFame()
        self.current_graph = self.current_quality = self.current_plan = None

    def optimize_plan(self, graph, t, threads=5):
        self.current_graph = graph
        self.current_plan = None
        self.current_quality = float("inf"), float("inf")

        ctl = Control([f"-ct={t}", "-Wno-operation-undefined", "--opt-mode=opt", f"-t{threads}"])
        # -Wno-operation-undefined is used to disable info: tuple ignored:  #sup@0
        # TODO - get better understanding why the warning occurs. It probably has to do with the situation when no plan exists.
        ctl.load("planner_2d.lp")
        # TODO - use knowledge about current best plan to constraint new ones for faster refutation
        program = model_to_program(graph) #+ self.models.constraint_to_better()
        ctl.add("base", (), program)
        ctl.ground([("base",())])

        result = ctl.solve(on_model=self.on_plan_model)
        assert result.exhausted

        if result.satisfiable:
            joint_model = self.current_graph, self.current_plan
            self.models.add(joint_model, self.current_quality)

    def on_plan_model(self, model):
        quality = tuple(model.cost)

        #if self.current_quality > quality:
        assert self.current_quality > quality # Something is wrong if not. Or I changed --opt-mode
        self.current_quality = quality
        self.current_plan = model.symbols(shown=True)
        # TODO - learn how to make model.optimality_proven actually work

    @property
    def best(self):
        return max(self.models.items(), key=itemgetter(1))


class Generator:
    # TODO (DIFFICULT!) get rid of redundant symmetric models
    # or prove they really need to be accepted.

    def __init__(self, args, max_time=9):
        self.args = args
        self.max_time = max_time

        self.planner = Planner()

    def on_model(self, model):
        symbols = model.symbols(shown=True)
        self.planner.optimize_plan(symbols, max_time)

    def run(self, topology):
        args_str = "_".join(f"{k}={v}" for k,v in sorted(self.args.items()))
        output_directory = f"by_maxtime/{self.max_time}/{args_str}"
        output_file_name = f"{output_directory}/{topology}"
        if os.path.isfile(output_file_name):
            print(f"file {output_file_name} already exists, skipping")
            return
        print("attempting", output_file_name)

        args = ["-n0"]
        for k, v in self.args.items():
            args.append(f"-c{k}={v}")

        ctl = Control(args)
        ctl.load(f"topology/{topology}.lp")
        ctl.load("connectivity.lp")
        ctl.ground([("base",())])

        sol = ctl.solve(on_model=self.on_model)
        assert sol.exhausted

        (graph, plan), cost = self.planner.models.best
        output_data = Graph2dSquare.fromModel(graph+plan).formatPlanMathematica()
        os.makedirs(output_directory, exist_ok=True)
        with open(output_file_name, "w") as outf:
            print(output_data, file=outf)
            print(cost, file=outf)
        print("wrote file", output_file_name)
        with open(output_file_name + "_model", "w") as outf:
            for atom in graph:
                print(str(atom)+".", file=outf)
        print("wrote file", output_file_name+"_model")


if __name__ == "__main__":
    for max_time in range(2, 10):
        for size in [2,3,4]:
            #max_components = size**2 // 2 + 1 # kinda arbitrary. Perhaps too low or too hight?
            max_components = size+1
            for n_components in range(3, max_components):
                args_dict = {"x": size, "y": size, "k": n_components}
                generator  = Generator(args_dict, max_time)
                generator.run("basic_2d_grid")
