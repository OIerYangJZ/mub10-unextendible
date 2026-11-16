#!/usr/bin/env python
"""Convert an msolve input file (vars / characteristic / polys) to a Singular script that prints the reduced
standard basis (std) and, with --lift, the cofactor polynomials h with 1 = sum_v h_v g_v (one per line).
Usage: python ms2sing.py <file.ms> <out.sing> [--lift <cofactor_out>]"""
import sys
src, dst = sys.argv[1], sys.argv[2]
lines = open(src).read().split("\n")
V, p = lines[0].strip(), int(lines[1])
polys = [q.strip() for q in "\n".join(lines[2:]).strip().split(",\n") if q.strip()]
s = ["ring R = %d, (%s), dp;" % (p, V), "option(redSB);", "ideal I = " + ",\n  ".join(polys) + ";",
     "ideal G = std(I);", 'print("STD_BEGIN");', "print(G);", 'print("STD_END");']
if "--lift" in sys.argv:
    out = sys.argv[sys.argv.index("--lift") + 1]
    s += ["matrix T = lift(I, ideal(1));", 'link L = ":w %s";' % out,
          "int k; for (k = 1; k <= ncols(I); k++) { write(L, string(T[k,1])); }", "close(L);", 'print("LIFT_DONE");']
s.append("quit;")
open(dst, "w").write("\n".join(s) + "\n")
