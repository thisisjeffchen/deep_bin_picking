import random


class LinearSchedule(object):
    def __init__(self, eps_begin, eps_end, nsteps):
        """
        Args:
            eps_begin: initial exploration
            eps_end: end exploration
            nsteps: number of steps between the two values of eps
        """
        self.epsilon        = eps_begin
        self.eps_begin      = eps_begin
        self.eps_end        = eps_end
        self.nsteps         = nsteps
    def update(self, t):
        if t == 0:
            self.epsilon = self.eps_begin
        elif t >= self.nsteps:
            self.epsilon = self.eps_end
        else:
            self.epsilon = self.eps_begin - t * 1.0 * (self.eps_begin - self.eps_end) / self.nsteps

class LinearExploration(LinearSchedule):
    def __init__(self, env, eps_begin, eps_end, nsteps):
        """
        Args:
            env: gym environment
            eps_begin: initial exploration
            eps_end: end exploration
            nsteps: number of steps between the two values of eps
        """
        self.env = env
        super(LinearExploration, self).__init__(eps_begin, eps_end, nsteps)


    def get_action(self, best_action):
        r = random.random ()

        if (r < self.epsilon):
            #TODO: make this work for our case, because we cannot sample
            #return self.env.action_space.sample ()
            raise NotImplementedError            

        else:
            return best_action

if __name__ == "__main__":
    ls = LinearSchedule (0, 10, 5)
    le = LinearExploration (None, 0, 10, 5)
    print "Init works!"