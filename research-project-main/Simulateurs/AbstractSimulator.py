from abc import ABC, abstractmethod
from typing import List, Any,Tuple

class AbstractSimulator(ABC):
    """Base class for all simulators."""

    @abstractmethod
    def get_legal_actions(self,state: Any) -> List[Any]:
        """
        Returns the possible actions from the current state of the simulator.

        Args:
            state (Any): The currently state.
            
        Returns:
            List[Any]: A list of legal actions available from the current state.
        """
        pass

    @abstractmethod
    def take_action(self, action: Any, player: Any) -> Any:
        """
        Executes an action in the simulator.

        Args:
            action (Any): The action to execute.
            player (Any): The player performing the action.

        Returns:
            Any: The new state of the simulator after the action is executed.
        """
        pass

    @abstractmethod
    def get_state(self) -> Any:
        """
        Returns the current state of the simulator.

        Returns:
            Any: The current state of the simulator.
        """
        pass

    @abstractmethod
    def victory(self, player: Any) -> bool:
        """
        Determines if the game is in a terminal state (victory, defeat, or draw).

        Args:
            player (Any): The player who made the action.

        Returns:
            bool: True if the game is won, False otherwise.
        """
        pass
    
    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, int, bool]:
        """
        Executes an action in the simulator and returns the resulting state, reward, and done flag.
        If the simulator supports multiple players, this function also plays the adversary's action.

        Args:
            action (Any): The action taken by the player.

        Returns:
            Tuple[Any, int, bool]: A tuple containing:
                - state (Any): The future state after the action.
                - reward (int): The reward obtained for this move.
                - done (bool): Whether the state is terminal.
        """
        pass