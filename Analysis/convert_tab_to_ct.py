import sys

tab_file = sys.argv[1]
ct_file = sys.argv[2]

with open(tab_file) as tab, open(ct_file, "w") as ct:
    ct.write(tab.readline())  # header
    for line in tab:
        index, base, partner = line.strip().split()
        index = int(index)
        partner = int(partner)
        ct.write(f"{index} {base} {index - 1} {index + 1} {partner} {index}\n")

