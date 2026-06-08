from dataclasses import dataclass


@dataclass
class TestCase:

    question: str
    category: str
    reference_answer: str
    keywords: list[str]


TEST_CASES = [

    TestCase(
        question="Who founded the company?",
        category="company",
        reference_answer="The company was founded by John Smith.",
        keywords=[
            "John Smith",
            "founder"
        ]
    ),

    TestCase(
        question="What products does the company offer?",
        category="products",
        reference_answer="The company provides insurance products.",
        keywords=[
            "insurance",
            "products"
        ]
    ),

    TestCase(
        question="What employee benefits are available?",
        category="employees",
        reference_answer="Employees receive benefits.",
        keywords=[
            "benefits",
            "employees"
        ]
    ),

]

def load_tests():

    return TEST_CASES