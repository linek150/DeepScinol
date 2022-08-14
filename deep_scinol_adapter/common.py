M = "scinol_M"
S2 = "scinol_S2"
G = "scinol_G"
ETA = "scinol_eta"
P0 = "scinol_p0"
T = 't'

def get_param_state(module, p_name):
    p_state = dict()
    p_state[G] = module.get_buffer(p_name + G)
    p_state[S2] = module.get_buffer(p_name + S2)
    p_state[M] = module.get_buffer(p_name + M)
    p_state[ETA] = module.get_buffer(p_name + ETA)
    p_state[P0] = module.get_buffer(p_name + P0)
    p_state[T] = module.get_buffer(T)
    return p_state