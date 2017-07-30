@echo off
set eval_iter=8000

:eval
cd util
call python evaluation.py --iter=%eval_iter%
cd ..
pause