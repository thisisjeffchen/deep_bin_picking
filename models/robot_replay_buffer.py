import numpy as np
import random

def sample_n_unique(sampling_f, n):
    """Helper function. Given a function `sampling_f` that returns
    comparable objects, sample n such unique objects.
    """
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res

class RobotReplayBuffer(object):
    """
    Taken from Berkeley's Assignment
    """
    def __init__(self, size, frame_history_len=1):
        """This is a memory efficient implementation of the replay buffer.

        The sepecific memory optimizations use here are:
            - only store each frame once rather than k times
              even if every observation normally consists of k last frames
            - store frames as np.uint8 (actually it is most time-performance
              to cast them back to float32 on GPU to minimize memory transfer
              time)
            - store frame_t and frame_(t+1) in the same buffer.

        For the tipical use case in Atari Deep RL buffer with 1M frames the total
        memory footprint of this buffer is 10^6 * 84 * 84 bytes ~= 7 gigabytes

        Warning! Assumes that returning frame of zeros at the beginning
        of the episode, when there is less frames than `frame_history_len`,
        is acceptable.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        frame_history_len: int
            Number of memories to be retried for each observation.
        """
        self.size = size
        self.frame_history_len = frame_history_len

        self.next_idx      = 0
        self.num_in_buffer = 0

        self.obs = None
        self.actions_taken   = None
        self.action_choices = None
        self.action_choices_mask = None
        self.rewards   = None
        self.done     = None
        
    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size + 1 <= self.num_in_buffer
      
    def _encode_sample(self, idxes):
        #print "idxes"
        #print idxes
        #print "num in buffer"
        #print self.num_in_buffer
        obs_batch      = self.obs[idxes]
        act_batch      = self.actions_taken[idxes]
        rew_batch      = self.rewards[idxes]
        #TODO: this is a hack to make the sampling go back to the front
        next_obs_batch = np.concatenate([[self.obs[(idx + 1) % self.num_in_buffer]] for idx in idxes], 0)
        done_mask      = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)

        act_choices_batch = self.action_choices[idxes] 
        act_choices_mask = self.action_choices_mask[idxes]

        return obs_batch, act_choices_batch, act_choices_mask, act_batch, rew_batch, done_mask, next_obs_batch
      
    def sample(self, batch_size):
        """Sample `batch_size` different transitions.

        i-th sample transition is the following:

        when observing `obs_batch[i]`, action `act_batch[i]` was taken,
        after which reward `rew_batch[i]` was received and subsequent
        observation  next_obs_batch[i] was observed, unless the epsiode
        was done which is represented by `done_mask[i]` which is equal
        to 1 if episode has ended as a result of that action.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        act_batch: np.array
            Array of shape (batch_size,) and dtype np.int32
        rew_batch: np.array
            Array of shape (batch_size,) and dtype np.float32
        next_obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        done_mask: np.array
            Array of shape (batch_size,) and dtype np.float32
        """
        assert self.can_sample(batch_size)
        idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
        return self._encode_sample(idxes)
      
    def store_frame(self, frame, action_choices, action_choices_mask):
        """Store a single frame in the buffer at the next available index, overwriting
        old frames if necessary.

        Parameters
        ----------
        frame: np.array
            Array of shape (img_h, img_w, img_c) and dtype np.uint8
            the frame to be stored

        Returns
        -------
        idx: int
            Index at which the frame is stored. To be used for `store_effect` later.
        """
        if self.obs is None:
            self.obs      =  np.zeros([self.size] + list(frame.shape), dtype=np.float32)
            assert len(action_choices.shape) == 2 #code depends on this

            self.actions_taken   =  np.zeros([self.size, action_choices.shape[1]], dtype=np.float32)
            self.action_choices = np.zeros([self.size] + list(action_choices.shape), dtype=np.float32)            

            self.action_choices_mask = np.zeros ([self.size, action_choices.shape[0]], dtype=np.bool)
            self.rewards = np.empty([self.size], dtype=np.float32)
            self.done = np.empty([self.size], dtype=np.bool)
        self.obs[self.next_idx] = frame
        self.action_choices[self.next_idx] = action_choices
        self.action_choices_mask[self.next_idx] = action_choices_mask

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret
      
    def store_effect(self, idx, action, reward, done):
        """Store effects of action taken after obeserving frame stored
        at index idx. The reason `store_frame` and `store_effect` is broken
        up into two functions is so that once can call `encode_recent_observation`
        in between.

        Paramters
        ---------
        idx: int
            Index in buffer of recently observed frame (returned by `store_frame`).
        action: int
            Action that was performed upon observing this frame.
        reward: float
            Reward that was received when the actions was performed.
        done: bool
            True if episode was finished after performing that action.
        """
        self.actions_taken[idx] = action
        self.rewards[idx] = reward
        self.done[idx]   = done
                                     
    #TODO: add a flushing function to flush replay buffer because we need the training data since our sim runs so slowly
    #Or maybe we can save the model params? but what if we change the model?


if __name__ == "__main__":
    rrb = RobotReplayBuffer (100)
    print "Init works!"

    #more tests
    state = np.ones (100)
    action_choices = np.ones ([5, 4])
    action_choices_mask = np.ones (5, dtype=bool)

    idx = rrb.store_frame (state, action_choices, action_choices_mask)

    action_taken = action_choices[0]
    reward = 5
    done = False

    rrb.store_effect (idx, action_taken, reward, done)

    state_prime = -1 * np.ones (100)
    action_choices_prime = -1 * np.ones ([5, 4])
    action_choices_prime_mask = -1 * np.ones (5, dtype = bool)

    idx2 = rrb.store_frame (state_prime, action_choices_prime, action_choices_prime_mask)

    print "Assert storage"
    assert rrb.action_choices[0][0][0] == action_choices[0][0]
    assert rrb.obs[0][0] == state[0]
    assert rrb.rewards[0] == reward

    print "Storage seems OK!"


    size = 1
    obs_batch, act_choices_batch, act_choices_mask, act_batch, rew_batch, done_mask, next_obs_batch = rrb.sample (size)

    print "Test retrieval"
    assert obs_batch.shape == (size, 100)
    assert act_choices_batch.shape == (size, 5, 4)
    assert act_choices_mask.shape == (size, 5)
    assert act_batch.shape == (size, 4)
    assert rew_batch.shape == (size,)
    assert next_obs_batch.shape == (size, 100)
    assert done_mask.shape == (size,)

    print "Sizes match!"

    print "Test Values"
    assert obs_batch[0][0] == 1
    assert act_choices_batch[0][0][0] == 1
    assert act_choices_mask[0][0] == 1
    assert act_batch[0][0]==1
    assert rew_batch[0] ==5
    assert next_obs_batch[0][0] == -1
    assert done_mask[0] == 0.0

    print "Values match!"





















