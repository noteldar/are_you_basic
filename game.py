import json
import random
import os
import time
import signal
from cost_function import evaluate_text


def clear_screen():
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def load_questions():
    """Load questions from questions.json file."""
    try:
        with open("questions.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # If file doesn't exist, create a default set of questions
        questions = [
            "What's your favorite book?",
            "What's your favorite movie?",
            "What's a hobby you enjoy?",
            "What's your favorite food?",
            "Where would you like to travel?",
            "What's your dream job?",
            "What's something you're proud of?",
            "What's a skill you'd like to learn?",
            "What's your favorite season and why?",
            "What's your favorite way to relax?",
        ]
        with open("questions.json", "w") as f:
            json.dump(questions, f)
        return questions


def display_intro():
    """Display game introduction."""
    clear_screen()
    print("=" * 50)
    print("           A R E  Y O U  B A S I C ?")
    print("=" * 50)
    print("\nProve your humanity by giving non-basic responses.")
    print("If you sound like AI or give nonsense, you're BASIC!")
    print("\nRules:")
    print("1. Each round costs $1 to play")
    print("2. Win $10 for your first non-basic response")
    print("3. Win $20 for your second consecutive win")
    print("4. Win $50 for your third consecutive win or more")
    print("5. If your response is basic, you lose your consecutive wins")
    print("6. You have 15 seconds to answer each question")
    print("7. Game over when you run out of money")
    print("\nLet's test if you're BASIC...")
    print("=" * 50)
    input("\nPress Enter to start...")


class TimeoutException(Exception):
    """Exception raised when timeout occurs."""

    pass


def timeout_handler(signum, frame):
    """Handler for timeout signal."""
    raise TimeoutException("Time's up!")


def timed_input(prompt, timeout=15):
    """Get input with a timeout."""
    # Print the prompt
    print(f"{prompt} (You have {timeout} seconds)")

    # Set up the timeout signal
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        user_input = input("Your response: ")
        # Cancel the alarm
        signal.alarm(0)
        return user_input
    except TimeoutException:
        # Cancel the alarm
        signal.alarm(0)
        print("\nTime's up!")
        return None


def main():
    # Initialize game
    display_intro()
    bank = 10
    consecutive_wins = 0
    questions = load_questions()

    # Main game loop
    while bank > 0:
        clear_screen()
        print(f"Bank: ${bank}")
        print(f"Consecutive wins: {consecutive_wins}")
        print("=" * 50)

        # Deduct cost to play
        bank -= 1

        # Select random question
        question = random.choice(questions)
        print(f"\nQuestion: {question}")

        # Get user response with timeout
        user_response = timed_input("Answer now", 15)

        # Handle timeout or empty response
        if user_response is None or not user_response.strip():
            consecutive_wins = 0  # Reset consecutive wins
            print("\n" + "=" * 50)
            if user_response is None:
                print("TIME'S UP! TOO SLOW!")
            else:
                print("EMPTY RESPONSE! TOO BASIC!")
            print("=" * 50)
            print(f"You lose this round. Bank: ${bank}")
            input("\nPress Enter to continue...")
            continue

        # Evaluate the response
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": user_response},
        ]

        results = evaluate_text(conversation)

        # Display results
        print("\n" + "=" * 50)
        print(f"Final Score: {results['final_score']:.3f}")
        print(f"AI Detection Score: {results['ai_detection_score']:.3f}")
        print(f"Coherence Score: {results['coherence_score']:.3f}")

        # Determine outcome
        if results["final_score"] >= 0.5:
            consecutive_wins += 1

            # Calculate winnings based on consecutive wins
            if consecutive_wins == 1:
                winnings = 10
            elif consecutive_wins == 2:
                winnings = 20
            else:  # 3 or more
                winnings = 50

            bank += winnings
            print(f"\nNot basic! You win ${winnings}!")

        else:
            consecutive_wins = 0  # Reset consecutive wins
            print("\n" + "=" * 50)
            print("Y O U  A R E  B A S I C !")
            print("=" * 50)
            print(f"You lose this round. Bank: ${bank}")

        input("\nPress Enter to continue...")

    # Game over when bank reaches 0
    print("\nYou've run out of money!")
    print("Game over!")
    print("\nThanks for playing!")


if __name__ == "__main__":
    main()
