import sys

df1_path = sys.argv[1]
df2_path = sys.argv[2]

import pandas
df1 = pandas.read_csv(df1_path)
df2 = pandas.read_csv(df2_path)

df1 = df1[["abstract_id", "copy_from", "copy_to"]]
df2 = df2[["abstract_id", "copy_from", "copy_to"]]

df1 = df1.sort_values(by=["abstract_id", "copy_from", "copy_to"], ignore_index=True, axis="rows")
df2 = df2.sort_values(by=["abstract_id", "copy_from", "copy_to"], ignore_index=True, axis="rows")

print(df1.equals(df2))