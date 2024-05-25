import pandas as pd

def apply_mutations_to_sequence(sequence, mutations_str):
        
        if mutations_str == "WT" or not mutations_str:
            # 如果没有突变（即野生型），直接返回原始序列
            return sequence
        mutations = mutations_str.split(';')
        sequence = list(sequence)  # 转换成列表便于修改
        for mut in mutations:
            if mut:  # 确保突变不为空
                original_aa, position, new_aa = mut[0], int(mut[1:-1]) - 1, mut[-1]
                assert sequence[position] == original_aa, f"Mutation at position {position+1} does not match the original amino acid."
                sequence[position] = new_aa
        return ''.join(sequence)
    

        
protein_sequence = """MRRESLLVSVCKGLRVHVERVGQDPGRSTVMLVNGAMATTASFARTCKCLAEHFNVVLFDLPFAGQSRQHNPQRGLITKDDEVEILLALIERFEVNHLVSASWGGISTLLALSRNPRGIRSSVVMAFAPGLNQAMLDYVGRAQALIELDDKSAIGHLLNETVGKYLPQRLKASNHQHMASLATGEYEQARFHIDQVLALNDRGYLACLERIQSHVHFINGSWDEYTTAEDARQFRDYLPHCSFSRVEGTGHFLDLESKLAAVRVHRALLEHLLKQPEPQRAERAAGFHEMAIGYA"""

annoation_file = pd.read_csv("proteinmul/test.csv")
save_dir = "./mutant_dataset/test"
for idx, row in annoation_file.iterrows():
    mutant_str = row["Mutant"]
    Activity = row.get("Activity", 0)
    Selectivity = row.get("Selectivity", 0)
    sequence = apply_mutations_to_sequence(protein_sequence, mutant_str)
    with open(f"{save_dir}/{idx}.fasta", "w") as f:
        f.write(f">{idx},{mutant_str}\n{sequence}\n{Activity};{Selectivity}\n")


# apply_mutations_to_sequence(protein_sequence, "WT")
