from gym_webots import Nao
nao = Nao(action_dim=6, obs_dim=22)
# nao.run()
# nao.reset()
for k in range(5):
    print(k)
    nao.run()
    nao.reset()