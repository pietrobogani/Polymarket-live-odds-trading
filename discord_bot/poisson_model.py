"""
Poisson Model for Football Match Probability Prediction

This module implements a Poisson-based model for predicting how probabilities
change when a team scores during a football match.

Supported markets:
- 1X2 (Home Win, Draw, Away Win)
- Over/Under 2.5 goals
- Handicap markets (e.g., Home -1.5, Home +1.5)

The model:
1. Takes current 1X2 market probabilities, score, and minute
2. Infers remaining expected goals (Λ_H, Λ_A) for each team using Poisson
3. Computes what all markets would become if either team scores

Mathematical basis:
- Goals scored by each team follow independent Poisson distributions
- The difference of two Poissons follows a Skellam distribution
- The sum of two Poissons is also Poisson (for O/U markets)
- We calibrate Λ values to match observed market probabilities
"""

import math
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List
from scipy.special import iv as bessel_i  # Modified Bessel function
from scipy.stats import poisson as poisson_dist
from scipy.optimize import minimize
import numpy as np


@dataclass
class MatchState:
    """Current state of a match."""
    minute: int  # Current minute (0-115 with our timing)
    home_goals: int
    away_goals: int

    @property
    def goal_diff(self) -> int:
        """Goal difference from home team's perspective."""
        return self.home_goals - self.away_goals


@dataclass
class Probabilities:
    """1X2 probabilities."""
    home_win: float
    draw: float
    away_win: float

    def normalize(self) -> 'Probabilities':
        """Normalize probabilities to sum to 1."""
        total = self.home_win + self.draw + self.away_win
        if total == 0:
            return Probabilities(1/3, 1/3, 1/3)
        return Probabilities(
            home_win=self.home_win / total,
            draw=self.draw / total,
            away_win=self.away_win / total
        )

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.home_win, self.draw, self.away_win)


@dataclass
class PoissonParams:
    """Poisson parameters for remaining goals."""
    lambda_home: float  # Expected remaining goals for home team
    lambda_away: float  # Expected remaining goals for away team


@dataclass
class GoalScenario:
    """Predicted probabilities after a hypothetical goal."""
    new_probs: Probabilities
    prob_increase_home: float  # Change in P(Home win)
    prob_increase_draw: float  # Change in P(Draw)
    prob_increase_away: float  # Change in P(Away win)


def skellam_pmf(k: int, lambda1: float, lambda2: float) -> float:
    """
    Probability mass function of the Skellam distribution.

    Skellam(k; λ1, λ2) = P(X - Y = k) where X ~ Poisson(λ1), Y ~ Poisson(λ2)

    Formula: e^{-(λ1+λ2)} * (λ1/λ2)^{k/2} * I_|k|(2*sqrt(λ1*λ2))

    Args:
        k: The value (can be negative)
        lambda1: Poisson parameter for first variable
        lambda2: Poisson parameter for second variable

    Returns:
        Probability P(X - Y = k)
    """
    if lambda1 <= 0 or lambda2 <= 0:
        # Edge case: if either lambda is 0, distribution is degenerate
        if lambda1 <= 0 and lambda2 <= 0:
            return 1.0 if k == 0 else 0.0
        elif lambda1 <= 0:
            # Only away team scores
            if k > 0:
                return 0.0
            # P(Y = -k) where Y ~ Poisson(lambda2)
            return math.exp(-lambda2) * (lambda2 ** (-k)) / math.factorial(-k) if k <= 0 else 0.0
        else:
            # Only home team scores
            if k < 0:
                return 0.0
            return math.exp(-lambda1) * (lambda1 ** k) / math.factorial(k) if k >= 0 else 0.0

    # Standard Skellam formula
    sqrt_prod = math.sqrt(lambda1 * lambda2)
    ratio = lambda1 / lambda2

    # Use log to avoid overflow
    log_prob = -(lambda1 + lambda2) + (k / 2) * math.log(ratio)
    bessel_val = bessel_i(abs(k), 2 * sqrt_prod)

    return math.exp(log_prob) * bessel_val


def compute_1x2_from_poisson(lambda_home: float, lambda_away: float,
                              current_diff: int, max_goals: int = 15) -> Probabilities:
    """
    Compute 1X2 probabilities given Poisson parameters and current score.

    Args:
        lambda_home: Expected remaining goals for home team
        lambda_away: Expected remaining goals for away team
        current_diff: Current goal difference (home - away)
        max_goals: Maximum goals to consider in summation

    Returns:
        Probabilities object with P(Home win), P(Draw), P(Away win)
    """
    p_home_win = 0.0
    p_draw = 0.0
    p_away_win = 0.0

    # Sum over possible goal differences from remaining play
    for additional_diff in range(-max_goals, max_goals + 1):
        prob = skellam_pmf(additional_diff, lambda_home, lambda_away)
        final_diff = current_diff + additional_diff

        if final_diff > 0:
            p_home_win += prob
        elif final_diff == 0:
            p_draw += prob
        else:
            p_away_win += prob

    return Probabilities(
        home_win=p_home_win,
        draw=p_draw,
        away_win=p_away_win
    )


def compute_over_under_probs(
    lambda_home: float,
    lambda_away: float,
    current_total: int,
    threshold: float = 2.5
) -> Tuple[float, float]:
    """
    Compute Over/Under probabilities given Poisson parameters.

    The sum of two independent Poissons is also Poisson:
    total_remaining ~ Poisson(lambda_home + lambda_away)

    Args:
        lambda_home: Expected remaining goals for home team
        lambda_away: Expected remaining goals for away team
        current_total: Current total goals scored
        threshold: O/U threshold (e.g., 2.5)

    Returns:
        Tuple of (P(Over), P(Under))
    """
    lambda_total = lambda_home + lambda_away

    # Goals needed to go over
    goals_needed = max(0, math.ceil(threshold) - current_total)

    if goals_needed <= 0:
        # Already over the threshold
        return (1.0, 0.0)

    if lambda_total <= 0.001:
        # No expected goals remaining
        return (0.0, 1.0)

    # P(Over) = P(remaining_goals >= goals_needed)
    # P(Under) = P(remaining_goals < goals_needed) = P(remaining_goals <= goals_needed - 1)
    p_under = poisson_dist.cdf(goals_needed - 1, lambda_total)
    p_over = 1 - p_under

    return (p_over, p_under)


def compute_handicap_prob(
    lambda_home: float,
    lambda_away: float,
    current_diff: int,
    handicap: float,
    for_home: bool = True
) -> float:
    """
    Compute handicap probability using Skellam distribution.

    Handicap betting asks: Will team beat the handicap?
    - Home -1.5: Home wins by 2+ goals → P(final_diff >= 2)
    - Home +1.5: Home doesn't lose by 2+ → P(final_diff >= -1)
    - Away -1.5: Away wins by 2+ → P(final_diff <= -2)

    Args:
        lambda_home: Expected remaining goals for home team
        lambda_away: Expected remaining goals for away team
        current_diff: Current goal difference (home - away)
        handicap: Handicap value (e.g., -1.5, +1.5)
        for_home: If True, calculate for home team handicap

    Returns:
        Probability of covering the handicap
    """
    # Threshold for goal difference to cover handicap
    # Home -1.5 means home needs diff >= 2 (wins by 2+)
    # Home +1.5 means home needs diff >= -1 (doesn't lose by 2+)
    if for_home:
        # Home team handicap: needs final_diff >= -handicap (rounded up)
        # Home -1.5 → need diff >= 2
        # Home +1.5 → need diff >= -1
        threshold = math.ceil(-handicap)
    else:
        # Away team handicap: needs final_diff <= handicap (rounded down)
        # Away -1.5 → need diff <= -2
        # Away +1.5 → need diff <= 1
        threshold = math.floor(handicap)

    # Sum probabilities using Skellam
    prob = 0.0
    max_goals = 15

    for additional_diff in range(-max_goals, max_goals + 1):
        p = skellam_pmf(additional_diff, lambda_home, lambda_away)
        final_diff = current_diff + additional_diff

        if for_home:
            if final_diff >= threshold:
                prob += p
        else:
            if final_diff <= threshold:
                prob += p

    return prob


@dataclass
class ExtendedMarketProbabilities:
    """Probabilities for all supported markets."""
    # 1X2
    home_win: float = 0.0
    draw: float = 0.0
    away_win: float = 0.0
    # Over/Under 2.5
    over_2_5: float = 0.0
    under_2_5: float = 0.0
    # Over/Under 1.5
    over_1_5: float = 0.0
    under_1_5: float = 0.0
    # Handicaps (home perspective)
    home_minus_1_5: float = 0.0  # Home -1.5 (home wins by 2+)
    home_plus_1_5: float = 0.0   # Home +1.5 (home doesn't lose by 2+)
    away_minus_1_5: float = 0.0  # Away -1.5 (away wins by 2+)
    away_plus_1_5: float = 0.0   # Away +1.5 (away doesn't lose by 2+)


def compute_all_market_probs(
    lambda_home: float,
    lambda_away: float,
    current_diff: int,
    current_total: int
) -> ExtendedMarketProbabilities:
    """
    Compute probabilities for all supported markets.

    Args:
        lambda_home: Expected remaining goals for home team
        lambda_away: Expected remaining goals for away team
        current_diff: Current goal difference (home - away)
        current_total: Current total goals

    Returns:
        ExtendedMarketProbabilities with all market probabilities
    """
    # 1X2
    probs_1x2 = compute_1x2_from_poisson(lambda_home, lambda_away, current_diff)

    # Over/Under 2.5
    over_2_5, under_2_5 = compute_over_under_probs(
        lambda_home, lambda_away, current_total, threshold=2.5
    )

    # Over/Under 1.5
    over_1_5, under_1_5 = compute_over_under_probs(
        lambda_home, lambda_away, current_total, threshold=1.5
    )

    # Handicaps
    home_minus_1_5 = compute_handicap_prob(
        lambda_home, lambda_away, current_diff, handicap=-1.5, for_home=True
    )
    home_plus_1_5 = compute_handicap_prob(
        lambda_home, lambda_away, current_diff, handicap=1.5, for_home=True
    )
    away_minus_1_5 = compute_handicap_prob(
        lambda_home, lambda_away, current_diff, handicap=-1.5, for_home=False
    )
    away_plus_1_5 = compute_handicap_prob(
        lambda_home, lambda_away, current_diff, handicap=1.5, for_home=False
    )

    return ExtendedMarketProbabilities(
        home_win=probs_1x2.home_win,
        draw=probs_1x2.draw,
        away_win=probs_1x2.away_win,
        over_2_5=over_2_5,
        under_2_5=under_2_5,
        over_1_5=over_1_5,
        under_1_5=under_1_5,
        home_minus_1_5=home_minus_1_5,
        home_plus_1_5=home_plus_1_5,
        away_minus_1_5=away_minus_1_5,
        away_plus_1_5=away_plus_1_5,
    )


def calibration_error(params: np.ndarray, target_probs: Probabilities,
                      current_diff: int) -> float:
    """
    Error function for calibration: difference between model and target probabilities.

    Args:
        params: [lambda_home, lambda_away]
        target_probs: Target probabilities from market
        current_diff: Current goal difference

    Returns:
        Sum of squared errors
    """
    lambda_home, lambda_away = params

    # Ensure positive lambdas
    if lambda_home < 0.001 or lambda_away < 0.001:
        return 1e10

    model_probs = compute_1x2_from_poisson(lambda_home, lambda_away, current_diff)

    error = (
        (model_probs.home_win - target_probs.home_win) ** 2 +
        (model_probs.draw - target_probs.draw) ** 2 +
        (model_probs.away_win - target_probs.away_win) ** 2
    )

    return error


def calibrate_poisson(target_probs: Probabilities, current_diff: int,
                      minutes_remaining: float) -> PoissonParams:
    """
    Calibrate Poisson parameters to match observed market probabilities.

    Args:
        target_probs: Observed market probabilities (normalized)
        current_diff: Current goal difference (home - away)
        minutes_remaining: Minutes remaining in the match (including stoppage)

    Returns:
        PoissonParams with calibrated lambda_home and lambda_away
    """
    # Initial guess based on typical goal rates
    # Average match has ~2.5 total goals, roughly even split
    avg_goals_per_90 = 2.5
    time_fraction = minutes_remaining / 90.0
    initial_total = avg_goals_per_90 * time_fraction

    # Adjust initial guess based on current probabilities
    if target_probs.home_win > target_probs.away_win:
        home_share = 0.55
    elif target_probs.away_win > target_probs.home_win:
        home_share = 0.45
    else:
        home_share = 0.50

    initial_lambda_home = initial_total * home_share
    initial_lambda_away = initial_total * (1 - home_share)

    # Ensure minimum values
    initial_lambda_home = max(0.01, initial_lambda_home)
    initial_lambda_away = max(0.01, initial_lambda_away)

    # Optimize
    result = minimize(
        calibration_error,
        x0=[initial_lambda_home, initial_lambda_away],
        args=(target_probs, current_diff),
        method='Nelder-Mead',
        options={'maxiter': 1000, 'xatol': 0.001, 'fatol': 0.0001}
    )

    lambda_home = max(0.001, result.x[0])
    lambda_away = max(0.001, result.x[1])

    return PoissonParams(lambda_home=lambda_home, lambda_away=lambda_away)


def predict_after_goal(current_probs: Probabilities, poisson_params: PoissonParams,
                       current_diff: int, home_scores: bool) -> GoalScenario:
    """
    Predict 1X2 probabilities after a hypothetical goal.

    Args:
        current_probs: Current 1X2 probabilities
        poisson_params: Calibrated Poisson parameters
        current_diff: Current goal difference (home - away)
        home_scores: True if home team scores, False if away team scores

    Returns:
        GoalScenario with new probabilities and changes
    """
    # New goal difference after the goal
    new_diff = current_diff + 1 if home_scores else current_diff - 1

    # Compute new probabilities with same lambda values
    new_probs = compute_1x2_from_poisson(
        poisson_params.lambda_home,
        poisson_params.lambda_away,
        new_diff
    )

    return GoalScenario(
        new_probs=new_probs,
        prob_increase_home=new_probs.home_win - current_probs.home_win,
        prob_increase_draw=new_probs.draw - current_probs.draw,
        prob_increase_away=new_probs.away_win - current_probs.away_win
    )


@dataclass
class ExtendedGoalScenario:
    """Predicted probabilities for all markets after a hypothetical goal."""
    new_probs: ExtendedMarketProbabilities
    prob_changes: Dict[str, float]  # market_key -> probability change


class PoissonPredictor:
    """
    Main class for predicting probability changes after goals.

    Supports all market types:
    - 1X2 (Home Win, Draw, Away Win)
    - Over/Under 2.5, 1.5 goals
    - Handicap markets (Home/Away ±1.5)

    Usage:
        predictor = PoissonPredictor()
        predictor.update(minute=30, home_goals=0, away_goals=0,
                        market_probs=Probabilities(0.5, 0.3, 0.2))

        decisions = predictor.get_buy_decisions(home_scores=True)
        # Returns decisions for ALL markets (1X2, O/U, handicaps)
    """

    # Match timing constants
    FIRST_HALF_END = 50  # 45 + 5 stoppage
    HALFTIME_END = 65    # 15 min break
    MATCH_END = 115      # 90 + 5 + 5 + 15 break = 115 total

    # Price constraints - configure for live trading
    MAX_PRICE = 0.98
    MIN_PROB_INCREASE = 0.05  # Minimum probability increase to trigger buy

    # Draw bias adjustment - calibrated from historical data
    DRAW_BIAS = 0.0  # Configure based on backtesting results

    def __init__(self):
        self.state: Optional[MatchState] = None
        self.current_probs: Optional[Probabilities] = None
        self.poisson_params: Optional[PoissonParams] = None
        self.home_goal_scenario: Optional[GoalScenario] = None
        self.away_goal_scenario: Optional[GoalScenario] = None
        # Extended market support
        self.current_extended_probs: Optional[ExtendedMarketProbabilities] = None
        self.home_goal_extended: Optional[ExtendedGoalScenario] = None
        self.away_goal_extended: Optional[ExtendedGoalScenario] = None

    def _get_minutes_remaining(self, minute: int) -> float:
        """
        Calculate effective minutes remaining in the match.

        Timeline:
        - 0-50: First half (45 min + 5 stoppage)
        - 50-65: Halftime (15 min, no play)
        - 65-115: Second half (45 min + 5 stoppage)
        """
        if minute < self.FIRST_HALF_END:
            # First half: remaining = (50 - minute) + 50 (second half)
            return (self.FIRST_HALF_END - minute) + 50
        elif minute < self.HALFTIME_END:
            # Halftime: 50 minutes of play remaining
            return 50
        else:
            # Second half: remaining = 115 - minute
            return max(0, self.MATCH_END - minute)

    def update(self, minute: int, home_goals: int, away_goals: int,
               market_probs: Probabilities) -> None:
        """
        Update the model with current match state and market probabilities.

        Args:
            minute: Current match minute (0-115 in our timing)
            home_goals: Current home team goals
            away_goals: Current away team goals
            market_probs: Current 1X2 probabilities from market
        """
        self.state = MatchState(minute=minute, home_goals=home_goals, away_goals=away_goals)
        self.current_probs = market_probs.normalize()

        # Get remaining time
        minutes_remaining = self._get_minutes_remaining(minute)

        # Calibrate Poisson parameters
        self.poisson_params = calibrate_poisson(
            self.current_probs,
            self.state.goal_diff,
            minutes_remaining
        )

        # Pre-compute scenarios (1X2 only - legacy)
        self.home_goal_scenario = predict_after_goal(
            self.current_probs,
            self.poisson_params,
            self.state.goal_diff,
            home_scores=True
        )

        self.away_goal_scenario = predict_after_goal(
            self.current_probs,
            self.poisson_params,
            self.state.goal_diff,
            home_scores=False
        )

        # Pre-compute extended market probabilities (O/U, handicaps)
        current_total = self.state.home_goals + self.state.away_goals
        self.current_extended_probs = compute_all_market_probs(
            self.poisson_params.lambda_home,
            self.poisson_params.lambda_away,
            self.state.goal_diff,
            current_total
        )

        # Pre-compute extended scenarios for both goal events
        self.home_goal_extended = self._compute_extended_scenario(home_scores=True)
        self.away_goal_extended = self._compute_extended_scenario(home_scores=False)

    def _compute_extended_scenario(self, home_scores: bool) -> ExtendedGoalScenario:
        """Compute extended market probabilities after a goal."""
        new_diff = self.state.goal_diff + (1 if home_scores else -1)
        new_total = self.state.home_goals + self.state.away_goals + 1

        new_probs = compute_all_market_probs(
            self.poisson_params.lambda_home,
            self.poisson_params.lambda_away,
            new_diff,
            new_total
        )

        # Calculate changes for each market
        current = self.current_extended_probs
        prob_changes = {
            'home_win': new_probs.home_win - current.home_win,
            'draw': new_probs.draw - current.draw,
            'away_win': new_probs.away_win - current.away_win,
            'over_2_5': new_probs.over_2_5 - current.over_2_5,
            'under_2_5': new_probs.under_2_5 - current.under_2_5,
            'over_1_5': new_probs.over_1_5 - current.over_1_5,
            'under_1_5': new_probs.under_1_5 - current.under_1_5,
            'home_minus_1_5': new_probs.home_minus_1_5 - current.home_minus_1_5,
            'home_plus_1_5': new_probs.home_plus_1_5 - current.home_plus_1_5,
            'away_minus_1_5': new_probs.away_minus_1_5 - current.away_minus_1_5,
            'away_plus_1_5': new_probs.away_plus_1_5 - current.away_plus_1_5,
        }

        return ExtendedGoalScenario(new_probs=new_probs, prob_changes=prob_changes)

    def get_buy_decisions(self, home_scores: bool) -> dict:
        """
        Get buy decisions for ALL markets after a goal.

        Returns dict with decisions for each market:
        - 'home_win': {'buy': bool, 'price_ceiling': float, 'probability_increase': float}
        - 'draw': {'buy': bool, 'price_ceiling': float, ...}
        - 'away_win': {'buy': bool, 'price_ceiling': float, ...}
        - 'over_2_5': {'buy': bool, 'price_ceiling': float, ...}
        - 'under_2_5': {'buy': bool, 'price_ceiling': float, ...}
        - 'home_minus_1_5': {'buy': bool, 'price_ceiling': float, ...}
        - etc.
        """
        extended = self.home_goal_extended if home_scores else self.away_goal_extended
        scenario = self.home_goal_scenario if home_scores else self.away_goal_scenario

        if extended is None or scenario is None:
            # Return empty decisions for all markets
            return {
                'home_win': {'buy': False, 'price_ceiling': 0, 'probability_increase': 0},
                'draw': {'buy': False, 'price_ceiling': 0, 'probability_increase': 0},
                'away_win': {'buy': False, 'price_ceiling': 0, 'probability_increase': 0},
                'over_2_5': {'buy': False, 'price_ceiling': 0, 'probability_increase': 0},
                'under_2_5': {'buy': False, 'price_ceiling': 0, 'probability_increase': 0},
                'over_1_5': {'buy': False, 'price_ceiling': 0, 'probability_increase': 0},
                'under_1_5': {'buy': False, 'price_ceiling': 0, 'probability_increase': 0},
                'home_minus_1_5': {'buy': False, 'price_ceiling': 0, 'probability_increase': 0},
                'home_plus_1_5': {'buy': False, 'price_ceiling': 0, 'probability_increase': 0},
                'away_minus_1_5': {'buy': False, 'price_ceiling': 0, 'probability_increase': 0},
                'away_plus_1_5': {'buy': False, 'price_ceiling': 0, 'probability_increase': 0},
            }

        decisions = {}
        new_probs = extended.new_probs
        changes = extended.prob_changes

        def make_decision(market_key: str, new_prob: float, bias: float = 0) -> dict:
            """Helper to create a buy decision for a market."""
            prob_increase = changes.get(market_key, 0)
            buy = prob_increase >= self.MIN_PROB_INCREASE
            price = min(self.MAX_PRICE, new_prob + bias) if buy else 0
            return {
                'buy': buy,
                'price_ceiling': price,
                'probability_increase': prob_increase,
            }

        # 1X2 Markets
        decisions['home_win'] = make_decision('home_win', new_probs.home_win)
        decisions['draw'] = make_decision('draw', new_probs.draw, bias=self.DRAW_BIAS)
        decisions['away_win'] = make_decision('away_win', new_probs.away_win)

        # Over/Under 2.5
        decisions['over_2_5'] = make_decision('over_2_5', new_probs.over_2_5)
        decisions['under_2_5'] = make_decision('under_2_5', new_probs.under_2_5)

        # Over/Under 1.5
        decisions['over_1_5'] = make_decision('over_1_5', new_probs.over_1_5)
        decisions['under_1_5'] = make_decision('under_1_5', new_probs.under_1_5)

        # Handicaps
        decisions['home_minus_1_5'] = make_decision('home_minus_1_5', new_probs.home_minus_1_5)
        decisions['home_plus_1_5'] = make_decision('home_plus_1_5', new_probs.home_plus_1_5)
        decisions['away_minus_1_5'] = make_decision('away_minus_1_5', new_probs.away_minus_1_5)
        decisions['away_plus_1_5'] = make_decision('away_plus_1_5', new_probs.away_plus_1_5)

        return decisions

    def get_state_summary(self) -> dict:
        """Get a summary of current model state for logging."""
        if self.state is None or self.current_probs is None:
            return {'initialized': False}

        return {
            'initialized': True,
            'minute': self.state.minute,
            'score': f"{self.state.home_goals}-{self.state.away_goals}",
            'current_probs': {
                'home': round(self.current_probs.home_win, 4),
                'draw': round(self.current_probs.draw, 4),
                'away': round(self.current_probs.away_win, 4)
            },
            'lambda_home': round(self.poisson_params.lambda_home, 4) if self.poisson_params else None,
            'lambda_away': round(self.poisson_params.lambda_away, 4) if self.poisson_params else None,
            'if_home_scores': {
                'home': round(self.home_goal_scenario.new_probs.home_win, 4),
                'draw': round(self.home_goal_scenario.new_probs.draw, 4),
                'away': round(self.home_goal_scenario.new_probs.away_win, 4),
                'changes': {
                    'home': f"{self.home_goal_scenario.prob_increase_home:+.2%}",
                    'draw': f"{self.home_goal_scenario.prob_increase_draw:+.2%}",
                    'away': f"{self.home_goal_scenario.prob_increase_away:+.2%}"
                }
            } if self.home_goal_scenario else None,
            'if_away_scores': {
                'home': round(self.away_goal_scenario.new_probs.home_win, 4),
                'draw': round(self.away_goal_scenario.new_probs.draw, 4),
                'away': round(self.away_goal_scenario.new_probs.away_win, 4),
                'changes': {
                    'home': f"{self.away_goal_scenario.prob_increase_home:+.2%}",
                    'draw': f"{self.away_goal_scenario.prob_increase_draw:+.2%}",
                    'away': f"{self.away_goal_scenario.prob_increase_away:+.2%}"
                }
            } if self.away_goal_scenario else None
        }


if __name__ == "__main__":
    # Example usage
    predictor = PoissonPredictor()
    market = Probabilities(home_win=0.50, draw=0.28, away_win=0.22)
    predictor.update(minute=30, home_goals=0, away_goals=0, market_probs=market)
    print(predictor.get_state_summary())
