#!/usr/bin/env python3
#coding=utf-8
"""Check a compressed Gaia DR3 CSV file for missing or invalid values.

Syntax:
  check_gdr3_csv.py csv_file

Output:
  A table showing, for each column, the number of:
    - empty strings
    - "null" strings
    - non‑convertible values (for numeric fields)
"""
import sys
import gzip
import csv
from io import StringIO
import numpy as np
from astrotoys.formats import gdr3_csv_dtype

def check_gdr3_csv(input_file):
    # Build a mapping from field name to numpy dtype kind
    field_kinds = {}
    for name, (dtype, _) in gdr3_csv_dtype.fields.items():
        field_kinds[name] = np.dtype(dtype).kind

    with gzip.open(input_file, 'rt') as f:
        # Skip comment lines (YAML)
        non_comment = [line for line in f if not line.startswith('#')]
    data = StringIO(''.join(non_comment))
    reader = csv.DictReader(data)
    fieldnames = reader.fieldnames

    stats = {name: {'total':0, 'empty':0, 'null':0, 'invalid':0}
             for name in fieldnames}

    for row in reader:
        for name in row:
            val = row[name]
            stats[name]['total'] += 1
            v = val.strip()
            if len(v) == 0:
                stats[name]['empty'] += 1
                continue
            if v.lower() == 'null':
                stats[name]['null'] += 1
                continue
            # Check numeric fields for convertibility
            kind = field_kinds.get(name, None)
            if kind is not None and kind in 'iuf':  # integer, unsigned, float
                try:
                    # Use float to test; int fields can be parsed as float too
                    float(v)
                except ValueError:
                    stats[name]['invalid'] += 1
            # Bool fields: valid values are 'true'/'false' (case-insensitive)
            elif kind == 'b':
                if v.lower() not in ('true', 'false'):
                    stats[name]['invalid'] += 1
            # String fields: no further validation required

    # Print results
    header = f"{'Field':<45} {'Total':>8} {'Empty':>8} {'Null':>8} {'Invalid':>8}"
    print(header)
    print('-' * len(header))
    for name in fieldnames:
        s = stats[name]
        if s['empty'] == 0 and s['null'] == 0 and s['invalid'] == 0:
            continue
        print(f"{name:<45} {s['total']:>8} {s['empty']:>8} {s['null']:>8} {s['invalid']:>8}")

    # Highlight phot_g_mean_mag if present
    if 'phot_g_mean_mag' in stats:
        s = stats['phot_g_mean_mag']
        print("\nphot_g_mean_mag details:")
        print(f"  Total rows: {s['total']}")
        print(f"  Empty: {s['empty']}, Null: {s['null']}, Invalid: {s['invalid']}")
        print(f"  Missing total (empty + null): {s['empty'] + s['null']}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: check_gdr3_csv.py <csv_file>")
        sys.exit(1)
    check_gdr3_csv(sys.argv[1])
