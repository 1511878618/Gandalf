try:
    from .pyg_mutant import (
        generate_nsSNP_pyg,
        get_mutantSequenceDict_from_chain_dict,
        mutantSequence_by_AAS,
        get_seq_list_from_chain_dict,
        get_sequence_from_Data,
        split_AAS,
    )
except (ImportError, NameError):
    pass