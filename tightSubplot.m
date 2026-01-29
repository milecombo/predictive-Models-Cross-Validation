function ax = tightSubplot(nr,nc,idx,hPad,vPad,rect)
% tightSubplot  Centered compact replacement for subplot (R2013b-safe)
% nr,nc : grid size
% idx   : subplot index (like subplot)
% hPad  : horizontal padding (left, right, and between columns) in normalized units
% vPad  : vertical padding (bottom, top, and between rows) in normalized units
% rect  : optional [x y w h] container region in normalized figure units (default ~full)

if nargin<4 || isempty(hPad), hPad = 0.02; end
if nargin<5 || isempty(vPad), vPad = 0.03; end
if nargin<6 || isempty(rect), rect = [0.04 0.05 0.92 0.90]; end  % roomy but centered

% subplot-style row/col (row counted from TOP)
row = ceil(idx/nc);
col = mod(idx-1,nc)+1;

% available space inside container
X0=rect(1); Y0=rect(2); W=rect(3); H=rect(4);

% compute axes width/height (symmetric margins + gaps)
w = (W - (nc+1)*hPad)/nc;
h = (H - (nr+1)*vPad)/nr;

% guard (prevents negative sizes if pads too large)
if w<=0 || h<=0, error('tightSubplot: pads too large for the chosen grid/rect.'); end

% position (CENTERED by construction: left/right and top/bottom margins are equal)
x = X0 + hPad + (col-1)*(w+hPad);
yTop = Y0 + H - vPad;                 % top inner margin reference
y = yTop - row*h - (row-1)*vPad;

ax = axes('Position',[x y w h]);
end