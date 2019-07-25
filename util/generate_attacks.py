#define adversarial attack
from advertorch.attacks import PGDAttack


def pgd_attack(model,loss_fn,eps=0.03,nb_iter=20,eps_iter=0.01,rand_init=True,targeted=False):
    adversary = PGDAttack(model,loss_fn=loss_fn,eps=eps,
                      nb_iter=nb_iter,eps_iter=eps_iter,rand_init=rand_init,clip_min=0.0,
                      clip_max=1.0,targeted=targeted)
    print('initial pgd Attacker')
    return adversary
