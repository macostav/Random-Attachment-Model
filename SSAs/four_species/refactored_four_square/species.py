species = [
    "A", "B", "C", "D",
    "AB", "AC", "BD", "CD",
    "ABD", "BCD", "ACD", "ABC",
    "ABCD"
]

n_species = len(species)
idx = {s: i for i, s in enumerate(species)} # Index

def n_particles(species_name):
    """Return the number of monomers in a species"""
    return len(species_name)