"""
Name lists for counterfactual (name-swap) resume auditing.

Source
------
Bertrand, M., & Mullainathan, S. (2004). "Are Emily and Greg More Employable
Than Lakisha and Jamal? A Field Experiment on Labor Market Discrimination."
American Economic Review, 94(4), 991-1013.

IMPORTANT: the lists below are an ABBREVIATED subset of the names used in the
paper. A real submission should use the COMPLETE published list from the
paper's appendix, not this shortened one, and verify every entry against it
(the last-name lists in particular).

Two attribute dimensions
------------------------
1. Name-associated group  -> NAME_GROUPS   ("groupA", "groupB")
2. Gender-associated first names -> GENDER_FIRST_NAMES ("female", "male")

Group keys are deliberately neutral labels. Their provenance is kept in each
group's "b_and_m_label" field:
    groupA = names the paper classifies as White-sounding
    groupB = names the paper classifies as African-American-sounding

Within each group, first names are split by gender so that an intersectional
label such as "female_groupB" maps to exactly one pool of first names.
GENDER_FIRST_NAMES is derived from those same pools (pooled across groups) and
is available for gender-only swaps.

These are name-based PROXIES used to build counterfactuals. They are not
ground-truth demographic attributes of any real applicant.
"""

NAME_GROUPS = {
    "groupA": {
        "b_and_m_label": "White-sounding",
        "first_names": {
            "female": ["Allison", "Anne", "Carrie", "Emily", "Jill",
                       "Laurie", "Kristen", "Meredith", "Sarah"],
            "male": ["Brad", "Brendan", "Geoffrey", "Greg", "Brett",
                     "Jay", "Matthew", "Neil", "Todd"],
        },
        "last_names": ["Baker", "Kelly", "McCarthy", "Murphy", "Murray",
                       "O'Brien", "Ryan", "Sullivan", "Walsh"],
    },
    "groupB": {
        "b_and_m_label": "African-American-sounding",
        "first_names": {
            "female": ["Aisha", "Ebony", "Keisha", "Kenya", "Lakisha",
                       "Latonya", "Latoya", "Tamika", "Tanisha"],
            "male": ["Darnell", "Hakim", "Jermaine", "Kareem", "Jamal",
                     "Leroy", "Rasheed", "Tremayne", "Tyrone"],
        },
        "last_names": ["Jackson", "Jones", "Robinson", "Washington",
                       "Williams", "Jefferson", "Booker"],
    },
}

GENDER_GROUPS = ["female", "male"]

# Gender dimension: pooled first names per gender, derived from NAME_GROUPS.
GENDER_FIRST_NAMES = {
    g: sorted({n for grp in NAME_GROUPS.values() for n in grp["first_names"][g]})
    for g in GENDER_GROUPS
}
