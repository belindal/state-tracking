from typing import List, Tuple
import numpy as np
import os
import json
import argparse
import itertools as it
from copy import deepcopy
from tqdm import tqdm
import random
from dataclasses import dataclass


@dataclass
class PermutationState:
    """
    Dataclass representing a permutation state.
    """
    permutation: Tuple[int, ...]
    
    def to_string(self) -> str:
        """Convert permutation to string representation."""
        return "".join([str(item) for item in self.permutation])
    
    @classmethod
    def from_string(cls, string_repr: str) -> 'PermutationState':
        """Create a PermutationState from its string representation."""
        permutation = tuple(int(char) for char in string_repr)
        return cls(permutation=permutation)
    
    def apply_action(self, action: Tuple[int, ...]) -> 'PermutationState':
        """
        Apply a permutation action to the current state.
        
        Args:
            action: Permutation action tuple
            
        Returns:
            new_state: The resulting state after applying the action
        """
        # Apply the permutation action to the current state
        new_perm = tuple(self.permutation[action[j]-1] for j in range(len(self.permutation)))
        return PermutationState(permutation=new_perm)


class PermutationTask:
    def __init__(self, num_items=5):
        """
        Initialize the permutation task.
        
        Args:
            num_items: Number of items (3 or 5)
        """
        self.num_items = num_items
        
        # Initialize all possible states and actions
        self.states = self._init_states()
        self.actions = self._init_actions()
        
        # Create mappings between string representations and objects
        self.state_to_nl = {state.permutation: state.to_string() for state in self.states}
        self.nl_to_state = {v: k for k, v in self.state_to_nl.items()}
        
        self.action_to_nl = {action.permutation: action.to_string() for action in self.actions}
        self.nl_to_action = {v: k for k, v in self.action_to_nl.items()}
        
        # Set initial state to identity permutation
        self.init_state = self.states[0]  # Identity permutation
        self.reset()
    
    def _init_states(self) -> List[PermutationState]:
        """
        Initialize all possible permutation states.
        
        Returns:
            states: List of all possible PermutationState objects
        """
        perms = list(it.permutations(range(1, self.num_items+1), self.num_items))
        return [PermutationState(permutation=perm) for perm in perms]
    
    def _init_actions(self) -> List[PermutationState]:
        """
        Initialize all possible permutation actions.
        
        Returns:
            actions: List of all possible PermutationState objects that can be used as actions
        """
        # For permutation composition, actions are also permutations
        return self._init_states()
    
    def get_valid_actions(self) -> List[PermutationState]:
        """
        Get list of valid actions from current state.
        
        Returns:
            valid_actions: List of valid PermutationState objects
        """
        # All actions are valid in the permutation task
        return self.actions
    
    def choose_random_action(self) -> PermutationState:
        """
        Choose a random valid action.
        
        Returns:
            action: Randomly chosen valid action
        """
        valid_actions = self.get_valid_actions()
        return random.choice(valid_actions)
    
    def action_to_natural_language(self, action: PermutationState) -> str:
        """
        Convert action to natural language.
        
        Args:
            action: PermutationState representing the action
            
        Returns:
            nl: Natural language representation of the action
        """
        return action.to_string()
    
    def reset(self) -> None:
        """
        Reset to initial state.
        """
        self.current_state = deepcopy(self.init_state)
    
    def update_state(self, action: PermutationState) -> None:
        """
        Update the current state based on the action taken.
        
        Args:
            action: PermutationState representing the action
        """
        self.current_state = self.current_state.apply_action(action.permutation)
    
    def simulate(self, steps=10, story_so_far="", states_so_far=None, num_stories=500, 
                write_dir="permutation_data", train_ratio=0.8) -> Tuple[List[str], List[List[PermutationState]]]:
        """
        Simulate the task for a given number of steps and stories.
        
        Args:
            steps: Number of steps per story
            story_so_far: Initial story text
            states_so_far: Initial states
            num_stories: Number of stories to generate
            write_dir: Directory to write stories to
            train_ratio: Ratio of train to total data
            
        Returns:
            stories: List of generated stories
            states: List of state sequences
        """
        if states_so_far is None:
            states_so_far = [self.init_state]
            
        stories = []
        states = []
        stories_set = set()
        
        os.makedirs(write_dir, exist_ok=True)
        train_dir = os.path.join(write_dir, "train")
        test_dir = os.path.join(write_dir, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        for story_idx in tqdm(range(num_stories), total=num_stories):
            self.reset()
            story_rollouts, state_rollouts = [], deepcopy(states_so_far)
            
            for _ in range(steps):
                action = self.choose_random_action()
                self.update_state(action)
                story_rollouts.append(self.action_to_natural_language(action))
                state_rollouts.append(deepcopy(self.current_state))
                
            story = story_so_far + " ".join(story_rollouts)
            if story in stories_set:
                continue
                
            stories.append(story)
            stories_set.add(story)
            states.append(state_rollouts)

            # Determine if this story goes to train or test set
            is_train = np.random.random() < train_ratio
            save_dir = train_dir if is_train else test_dir
            
            # Convert states to serializable format
            serializable_states = [state.permutation for state in state_rollouts]
            
            with open(f"{save_dir}/story_{story_idx}.json", "w") as f:
                f.write(json.dumps({
                    "story": story.strip(),
                    "state_seq": serializable_states,
                }, indent=4))
                
        return stories, states


def compute_parity(permutation, num_items=None) -> int:
    """
    Compute the parity of a permutation.
    
    Args:
        permutation: A permutation tuple or list
        num_items: Number of items (not used, kept for compatibility)
        
    Returns:
        parity: 0 for even permutations, 1 for odd permutations
    """
    # Convert to list if it's a tuple
    perm = list(permutation)
    
    # Count inversions
    inversions = 0
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            if perm[i] > perm[j]:
                inversions += 1
    
    # Return parity (0 for even, 1 for odd)
    return inversions % 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-items", type=int, default=3, choices=[3, 5], help="Number of items (3 or 5)")
    parser.add_argument("--story-length", type=int, default=100, help="Number of steps per story")
    parser.add_argument("--data-dir", type=str, default="S3_data", help="Directory to write stories to")
    parser.add_argument("--num-stories", type=int, default=1000000, help="Number of stories to generate")
    parser.add_argument("--train-ratio", type=float, default=0.999, help="Ratio of train to total data (e.g., 0.8 means 80% train, 20% test)")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    # Create task with specified number of items
    task = PermutationTask(num_items=args.num_items)
    
    # Simulate task
    stories, states = task.simulate(
        steps=args.story_length,
        num_stories=args.num_stories,
        story_so_far="",
        states_so_far=[task.init_state],
        write_dir=args.data_dir,
        train_ratio=args.train_ratio,
    )

    # Count how many stories went to train vs test
    train_count = len(os.listdir(os.path.join(args.data_dir, "train")))
    test_count = len(os.listdir(os.path.join(args.data_dir, "test")))
    
    print(f"Generated {len(stories)} stories with {args.num_items} items")
    print(f"Split: {train_count} train, {test_count} test (ratio: {train_count/(train_count+test_count):.2f})")
    print("Sample story:", stories[0] if stories else "No stories generated") 
