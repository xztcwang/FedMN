
def bits_vec_prepare(args_):
    modular_units_list = args_.units_list
    m_modular_layers = len(modular_units_list)
    bits_vec_list = []
    bits_vec_list.append(modular_units_list[0])
    for i in range(1, m_modular_layers):
        bits_vec_list.append(modular_units_list[i - 1] * modular_units_list[i])
    bits_vec_list.append(modular_units_list[m_modular_layers - 1])
    return bits_vec_list

