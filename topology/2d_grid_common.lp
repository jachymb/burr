shape(x, y).

% TODO - space points are not parametrizable. The padding size is thus very rigid.
space_point((-X..2*X-1, -Y..2*Y-1)) :- shape(X, Y).

adjacent((X1, Y1), (X2, Y2)) :-
    |X1 - X2| + |Y1 - Y2| = 1,
    space_point((X1, Y1)),
    space_point((X2, Y2)).

label((X,Y), Y*A + X) :-
    shape(A, _),
    point((X,Y)).
