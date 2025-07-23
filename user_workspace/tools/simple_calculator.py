"""
A tool to perform basic arithmetic calculations (addition, subtraction, multiplication, division) on a list of numbers.

Generated tool: Simple Calculator
Category: custom
Risk Level: safe
Created: 1753275598.9852386
"""

def calculate(operation, numbers):
    """Performs a calculation based on the provided operation and numbers.

    Args:
        operation (str): The arithmetic operation to perform.
        numbers (list): A list of numbers to operate on.

    Returns:
        float: The result of the calculation. Returns None if the operation is invalid or if division by zero is attempted.
    """
    if operation == 'addition':
        result = sum(numbers)
    elif operation == 'subtraction':
        if not numbers:
            return None  # Handle empty list case
        result = numbers[0]
        for num in numbers[1:]:
            result -= num
    elif operation == 'multiplication':
        result = 1
        for num in numbers:
            result *= num
    elif operation == 'division':
        if not numbers:
            return None #Handle empty list case
        result = numbers[0]
        for num in numbers[1:]:
            if num == 0:
                print("Error: Division by zero is not allowed.")
                return None
            result /= num
    else:
        print("Error: Invalid operation.")
        return None
    return result

# Tool parameters: {'operation': {'type': 'string', 'description': "The arithmetic operation to perform.  Valid values: 'addition', 'subtraction', 'multiplication', 'division'."}, 'numbers': {'type': 'list of integers', 'description': 'A list of numbers to perform the operation on.'}}
# Examples: ["calculate('addition', [1, 2, 3])", "calculate('subtraction', [5, 2, 1])", "calculate('multiplication', [2, 3, 4])", "calculate('division', [10, 2, 2])"]
# Dependencies: []
