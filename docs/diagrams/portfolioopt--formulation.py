#!/usr/bin/env python3
"""portfolioopt--formulation: how a PortfolioOpt.jl strategy is built and evaluated.

Message: a returns model (a distribution P or an ambiguity set calP), a measure rho over it and its role
(objective weight c_j or constraint set C_i) make one PortfolioFormulation; JuMP solves it as a conic model,
the market clears the decision w* with the realized returns, and the backtest moves to the next period.

Drawn with the diagram kit of the author's website (tools/diagrams/kit.py in
github.com/andrewrosemberg/AndrewRosemberg); set KIT to that folder. Notation of docs/src/theory/problem.md.

Wide: lane "Formulation" (P -> rho -> role -> problem), lane "Backtest" (conic solve -> clear -> returns,
returns -> P crosses the split): one clockwise loop.
Compact: the same loop as one column, the forecast edge returning along the right side.
"""
import os
import sys

KIT = os.environ.get('KIT', os.path.expanduser('~/Workspace/AndrewRosemberg/tools/diagrams'))
sys.path.insert(0, KIT)
from kit import Diagram  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def two(a, b):
    return r'$$\begin{gathered}%s\\ %s\end{gathered}$$' % (a, b)


def lab(p, x, y, tex, w=120, h=20, align='c'):
    """Free math label centred at (x, y), or starting at x (align='l'), or ending at x (align='r')."""
    if align == 'l':
        return p.text(None, x, y - h / 2.0, w, h, tex, left=True)
    x0 = x - w / 2.0 if align == 'c' else x - w
    return p.text(None, x0, y - h / 2.0, w, h, tex)


def e(p, s, t, exit, entry, pts=()):
    p.edge(s, t, 'value', exit=exit, entry=entry, points=pts)


SETB = two(r'\text{returns model}', r'P\ \text{or}\ \mathcal{P}')
MEAS = two(r'\text{measure}', r'\rho(w)')
ROLE = two(r'\text{role}', r'c_j\ \text{or}\ \mathcal{C}_i')
PROB = two(r'\max_{w\in\mathcal{X}}\ \sum_j c_j\,\rho_j(w)', r'\text{s.t.}\ \ \rho_i(w)\in\mathcal{C}_i')
SOLVE = r'\text{conic solve}'
CLEAR = two(r'\text{clear market}', r'r_{t+1}^{\top} w^*')
HIST = r'r_{1:t}'
JUMP = r'\textit{JuMP}\ \text{conic model}'

d = Diagram()

# ------------------------------------------------------------------ wide
w = d.page('Page-1')
W, SPL, H = 540, 150, 250
YA, YB = 75, 210
w.frame(0, 0, W, H)
w.line(0, SPL, W, SPL, dashed=True)
w.lane_title(0, 0, 'Formulation', inset=True)
w.lane_title(0, SPL, 'Backtest', inset=True)
w.box('set', 10, YA - 22, 100, 44, SETB, role='model')
w.box('meas', 140, YA - 22, 80, 44, MEAS, role='model')
w.box('role', 250, YA - 22, 90, 44, ROLE, role='model')
w.box('prob', 370, YA - 32, 160, 64, PROB, role='data')
for a, b in (('set', 'meas'), ('meas', 'role'), ('role', 'prob')):
    e(w, a, b, (1, 0.5), (0, 0.5))
w.store('hist', 65, YB - 25, 60, 50, HIST)
w.box('clear', 240, YB - 22, 110, 44, CLEAR, role='loss')
w.box('solve', 400, YB - 18, 100, 36, SOLVE, role='data')
e(w, 'prob', 'solve', (0.5, 1), (0.5, 0))
lab(w, 444, 125, JUMP, w=110, align='r')
e(w, 'solve', 'clear', (0, 0.5), (1, 0.5))
lab(w, 375, YB - 12, r'w^*', w=24)
e(w, 'clear', 'hist', (0, 0.5), (1, 0.5))
lab(w, 182, YB - 12, r'\text{next period}', w=90)
e(w, 'hist', 'set', (0.5, 0), (0.85, 1))
lab(w, 101, 120, r'\text{forecast}', w=60, align='l')

# ------------------------------------------------------------------ compact
c = d.page('compact')
CW, SPC, H2 = 266, 325, 560
X = 140
c.frame(0, 0, CW, H2)
c.line(0, SPC, CW, SPC, dashed=True)
c.lane_title(0, 0, 'Formulation', inset=True)
c.lane_title(0, SPC, 'Backtest', inset=True)
c.box('set', X - 50, 40, 100, 44, SETB, role='model')
c.box('meas', X - 50, 104, 100, 44, MEAS, role='model')
c.box('role', X - 50, 168, 100, 44, ROLE, role='model')
c.box('prob', X - 80, 236, 160, 64, PROB, role='data')
for a, b in (('set', 'meas'), ('meas', 'role'), ('role', 'prob')):
    e(c, a, b, (0.5, 1), (0.5, 0))
c.box('solve', X - 50, 362, 100, 36, SOLVE, role='data')
c.box('clear', X - 55, 422, 110, 44, CLEAR, role='loss')
c.store('hist', X - 30, 492, 60, 50, HIST)
e(c, 'prob', 'solve', (0.5, 1), (0.5, 0))
lab(c, X - 6, 314, JUMP, w=110, align='r')
e(c, 'solve', 'clear', (0.5, 1), (0.5, 0))
lab(c, X + 6, 410, r'w^*', w=24, align='l')
e(c, 'clear', 'hist', (0.5, 1), (0.5, 0))
lab(c, X + 6, 479, r'\text{next period}', w=90, align='l')
e(c, 'hist', 'set', (1, 0.5), (1, 0.5), [(CW - 16, 517), (CW - 16, 62)])
lab(c, CW - 22, 158, r'\text{forecast}', w=56, align='r')

d.write(os.path.join(HERE, 'portfolioopt--formulation.drawio'))
