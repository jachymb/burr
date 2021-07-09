n_components(k).
component(1 .. K) :- n_components(K).

% This is a simple approach that ignores possible symmetries in the graph
% hence resulting in redundant (symmetric) models.
0 {connected_adjacent(Point1, Point2)} 1 :- adjacent(Point1, Point2), point(Point1), point(Point2).
% The relation is symmetric since the graph is unoriented.
connected_adjacent(Point1, Point2) :- connected_adjacent(Point2, Point1). 

% Connectivity is the equivalence closure of adjacenct connectivity
connected(Point1, Point2) :- connected_adjacent(Point1, Point2).
connected(Point, Point) :- point(Point).                                                    % Reflexivity closure
connected(Point1, Point2) :- connected(Point1, Point3), connected_adjacent(Point3, Point2). % Transitivity closure
% Symmetry closure is implied from adjacent conntectivity symmetry. 
% connected(Point1, Point2) :- connected(Point2, Point1).

% If two adjacent points are connected, there should be an edge between them.
% This is meant to remove useless slits. Not sure if it's a sufficient condition to remove them all.
:- connected(Point1, Point2), adjacent(Point1, Point2), not connected_adjacent(Point1, Point2).

% Now assign each point to a connected component.
% We will check if points assigned to the same components are connected and points assigned to different components are not
% This seems very slow, but it seems impossible to implement Tarjan-like algorithm in ASP.
1 {belongs_to(Point, Component) : component(Component) } 1 :- point(Point).

% If two points belong to the same component, then they are connected
:- belongs_to(Point1, Component),
   belongs_to(Point2, Component),
   not connected(Point1, Point2).

% If two points are connected, then they belong to the same component
:- belongs_to(Point1, Component1),
   belongs_to(Point2, Component2),
   connected(Point1, Point2),
   Component1 != Component2.

% Determines the number of points in each component
component_size(Component, Size) :- 
    Size = #count{Point : belongs_to(Point, Component)},
    component(Component).

% There are no empty components
:- component_size(_, 0).

% The component_represenant serves like an AOC function for the connectivity equivalence classes
% It selects the lowest numbered label.
component_representant(Component, MinId) :-
    MinId = #min{Id : label(Point, Id), belongs_to(Point, Component)},
    component(Component).

% Symmetry reduction: make the component numbers follow the same order as their representants.
:- component_representant(Component1, Id1),
   component_representant(Component2, Id2),
   Component1 < Component2,
   Id1 >= Id2. 

% Extends belongs_to to labels instead of points.
label_belongs_to(Id, Component) :- label(Point, Id), belongs_to(Point, Component). 
