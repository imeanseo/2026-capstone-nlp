#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
분석 결과 정리 스크립트
P1 분석 후 생성된 파일들을 results/p1_analysis/ 폴더로 이동
"""

import os
import shutil
from pathlib import Path

print("="*80)
print("분석 결과 정리 중...")
print("="*80)

# 현재 디렉토리
base_dir = Path(__file__).parent.parent
results_dir = base_dir / "results" / "p1_analysis"

# P1 결과 폴더 생성
results_dir.mkdir(parents=True, exist_ok=True)

# 이동할 파일 목록
files_to_move = [
    'surface_cue_analysis.png',
    'borderline_강함.csv',
    'borderline_중간.csv',
    'borderline_약함.csv',
    'borderline_없음.csv',
    'template_cell_c.csv',
    'template_cell_d.csv',
    'template_cell_b.csv',
    'p1_analysis_summary.csv',
]

moved_count = 0
not_found = []

for filename in files_to_move:
    source = base_dir / filename
    if source.exists():
        dest = results_dir / filename
        shutil.move(str(source), str(dest))
        print(f"✅ {filename} → results/p1_analysis/")
        moved_count += 1
    else:
        not_found.append(filename)

print("\n" + "="*80)
print(f"✅ 정리 완료: {moved_count}개 파일 이동")

if not_found:
    print(f"\n⚠️ 찾을 수 없는 파일: {len(not_found)}개")
    for f in not_found:
        print(f"  - {f}")

print("\n📂 결과 폴더 구조:")
print(f"  results/p0_base/     : P0 분석 결과")
print(f"  results/p1_analysis/ : P1 분석 결과")
print("="*80)
