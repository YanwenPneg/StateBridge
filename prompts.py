# Portions adapted from LatentMAS (https://github.com/Gen-Verse/LatentMAS).
# Licensed under Apache-2.0 and modified by the StateBridge authors.
# See THIRD_PARTY_NOTICES.md.

"""
Prompt constructors for StateBridge multi-agent system.

Only the embedding-MAS prompt builder is included, which is used by the
StateBridge method for latent-state communication between agents.
"""

# Marker token indicating where embedding prefix should be inserted
EMBEDDING_HINT_MARKER = "[EMBEDDING_CONTEXT_HERE]"


def build_agent_message_embedding_mas(
    role: str,
    question: str,
    context: str = "",
    method=None,
    args=None,
    has_prefix: bool = False,
):
    """Build prompt messages for the StateBridge embedding-MAS method.

    For agents receiving an embedding prefix from a preceding agent,
    a marker (EMBEDDING_HINT_MARKER) is placed in the prompt text.
    The StateBridge class later replaces this marker with the actual
    embedding injection position.

    Args:
        role: Agent role (planner, critic, refiner, judger).
        question: The input question text.
        context: Unused, kept for API compatibility.
        method: Unused, kept for API compatibility.
        args: Namespace with at least `model` and `task` attributes.
        has_prefix: Whether this agent receives an embedding prefix
                    from a preceding agent.
    """

    # Model-agnostic system message
    model_name = getattr(args, 'model', '') or getattr(args, 'model_name', '') if args else ''
    if "qwen" in model_name.lower():
        base_system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    elif "olmo" in model_name.lower():
        base_system_message = "You are a helpful assistant."
    else:
        base_system_message = "You are a helpful assistant."

    system_message = base_system_message

    # Embedding context hint (only added when receiving a prefix; uses MARKER placeholder)
    if has_prefix:
        embedding_context_hint = f"The following is the message from previous Agent (provided in embedding format):\n{EMBEDDING_HINT_MARKER}\n\n"
    else:
        embedding_context_hint = ""

    if role == "planner":
        # Planner is the first agent; does not receive embeddings
        user_prompt = f"""You are a Planner Agent. Given an input question, design a clear, step-by-step plan for how to solve the question.

Question: {question}

Your outlined plan should be concise with a few bulletpoints for each step. Do not produce the final answer.
Now output your plan to solve the question below:
"""

    elif role == "critic":
        user_prompt = f"""{embedding_context_hint}Question: {question}

You are a Critic Agent to evaluate the correctness of the input plan for the given question and provide helpful feedback for improving the plan.
The plan information is provided in embedding representation format. Review the plan and question and output:
(1) original plan contents
(2) constructive feedback on the original plan.

Format your response as follows:
Original Plan: [Copy the provided Planner Agent's plan here]
Feedback: [Your detailed feedback to improve the plan here]

Now, output your response below:
"""

    elif role == "refiner":
        user_prompt = f"""{embedding_context_hint}Question: {question}

You are a Refiner Agent to provide a refined step-by-step plan for solving the given question.
You are provided with:
(1) embedding-format information: a previous plan with feedback
(2) text-format information: the input question you need to solve.

Based on the input, write a refined and improved plan to solve the question. Make sure your output plan is correct and concise.

Now, output your refined plan below:
"""

    elif role == "judger":
        if args.task in ['gsm8k', 'aime2024', 'aime2025']:
            user_prompt = f"""{embedding_context_hint}Target Question: {question}

You are a helpful assistant. You are provided with embedding information for reference and a target question to solve. 

The embedding information might contain irrelevant contents. Ignore it if it is not helpful for solving the target question.

You must reason step-by-step to solve the provided Target Question without outputting other irrelevant information.

Now, reason step by step and output the final answer inside \\\\boxed{{YOUR_FINAL_ANSWER}}.
"""

        elif args.task in ["arc_easy", "arc_challenge", "gpqa", 'medqa']:
            user_prompt = f"""{embedding_context_hint}Target Question: {question}

You are a helpful assistant. You are provided with embedding information for reference and a target question to solve. 

The embedding information might contain irrelevant contents. Ignore it if it is not helpful for solving the target question.

You must reason step-by-step to solve the provided Target Question without outputting other irrelevant information.
Your final answer must be selected from A,B,C,D. For example \\\\boxed{{A}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\\\boxed{{YOUR_FINAL_ANSWER}}.
"""

        elif args.task in ["mbppplus", "humanevalplus"]:
            user_prompt = f"""{embedding_context_hint}Target Question: {question}

You are a helpful assistant. You are provided with embedding information for reference and a target question to solve.

The embedding information might contain irrelevant contents. Ignore it if it is not helpful for solving the target question.

You must reason step-by-step to solve the provided Target Question without outputting other irrelevant information.
You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import math
def add(a, b):
    return a + b```. Do not add any other contents inside the markdown code block.

Now, reason step by step and output the final answer inside ```python
YOUR_PYTHON_CODE
```.
"""

        elif args.task in ["winogrande"]:
            user_prompt = f"""{embedding_context_hint}Target Question: {question}

You are a helpful assistant. You are provided with embedding information for reference and a target question to solve. 

The embedding information might contain irrelevant contents. Ignore it if it is not helpful for solving the target question.

You must reason step-by-step to solve the provided Target Question without outputting other irrelevant information.
Your final answer must be selected from 1 and 2. For example \\\\boxed{{1}} or \\\\boxed{{2}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\\\boxed{{YOUR_FINAL_ANSWER}}.
"""

        else:
            raise NotImplementedError(f"Task {args.task} not implemented in embedding_mas judger prompt.")

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]
