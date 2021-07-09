max_time(t).
timestamp(0..T-1) :- max_time(T).
action_param(-1;1).
action((move_x,P); (move_y,P)) :- action_param(P).

% At each step, perform one action at one component
1 {execute(Timestamp, Action, Component) : action(Action), component(Component)} 1 :- timestamp(Timestamp).

% Little speedup heuristic: Do not move the same piece back and forth
:- execute(Timestamp,   (Direction, Shift1), Component),
   execute(Timestamp+1, (Direction, Shift2), Component),
   Shift2 = -Shift1.

% Start with all points (represented by labels) not moved
at(0, Id, Point) :- label(Point, Id).
% Define movement of component in x direction
at(T+1, Id, Position) :-
    execute(T, (move_x, Shift), Component),
    label_belongs_to(Id, Component),
    at(T, Id, (X,Y)),
    Position = (X+Shift, Y),
    space_point(Position).
% Define movement of component in y direction
at(T+1, Id, Position) :-
    execute(T, (move_y, Shift), Component),
    label_belongs_to(Id, Component),
    at(T, Id, (X,Y)),
    Position = (X, Y+Shift),
    space_point(Position).
% Keep other components in place
at(T+1, Id, Position) :-
    execute(T, _, Component1),
    label_belongs_to(Id, Component2),
    at(T, Id, Position),
    Component1 != Component2.

% Different nodes may never occupy the same space
:- at(Timestamp, Id1, Position), at(Timestamp, Id2, Position), Id1 != Id2.

% Block adjacency at given step
adjacent_at(Timestamp, Id1, Id2) :-
    at(Timestamp, Id1, Position1),
    at(Timestamp, Id2, Position2),
    adjacent(Position1, Position2).

% Component adjacency at given step
component_adjacent_at(Timestamp, Component1, Component2) :-
    adjacent_at(Timestamp, Id1, Id2),
    label_belongs_to(Id1, Component1),
    label_belongs_to(Id2, Component2),
    Component1 != Component2.

plan_complexity(0, 1).
plan_complexity(Timestamp+1, Complexity) :-
    execute(Timestamp, (Direction, _), Component),
    execute(Timestamp+1, (Direction, _), Component),
    plan_complexity(Timestamp, Complexity).
plan_complexity(Timestamp+1, Complexity+1) :-
    execute(Timestamp, (Direction1, _), _),
    execute(Timestamp+1, (Direction2, _), _),
    Direction1 != Direction2,
    plan_complexity(Timestamp, Complexity).
plan_complexity(Timestamp+1, Complexity+1) :-
    execute(Timestamp, _, Component1),
    execute(Timestamp+1, _, Component2),
    plan_complexity(Timestamp, Complexity),
    component(Component1), component(Component2),
    Component1 != Component2.

% Define when a valid plan is found
solved_at(Timestamp) :-
    AdjacentComponents = #count{Component : component_adjacent_at(Timestamp, Component, _) },
    AdjacentComponents = 0,
    timestamp(Timestamp-1).

% Only interested in the first solution. Everything beyond is redundant.
min_solved_at(Timestamp) :- Timestamp = #min{ T : solved_at(T) }.

solution_complexity(Complexity) :- min_solved_at(Timestamp), plan_complexity(Timestamp-1, Complexity).

% Optimization directive
:~ solution_complexity(Complexity). [Complexity@0]
:~ min_solved_at(Timestamp). [Timestamp@1]

% Require plan sovability
:- not solved_at(_).

#show execute/3.
#show solved_at/1.
#show at/3.
#show belongs_to/2.
#show max_time/1.
#show min_solved_at/1.
#show solution_complexity/1.
#show plan_complexity/2.
