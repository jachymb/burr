% The boundary points:
point((0, 0 .. Y-1)) :- shape(_, Y).
point((X-1, 0 .. Y-1)) :- shape(X, Y).
point((0 .. X-1, 0)) :- shape(X, _).
point((0 .. X-1, Y-1)) :- shape(X, Y).

% The inner region:
{point((1 .. X-2, 1 .. Y-2))} :- shape(X, Y).
%FIXME - for larger shapes, this can create disconnected components inside.
%But perhaps that can be ignored?

#include "2d_grid_common.lp".
