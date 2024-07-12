import subprocess

#
# Query CLI
# The GraphRAG query CLI allows for no-code usage of the GraphRAG Query engine.
# python -m graphrag.query --data <path-to-data> --community_level <community-level> --response_type <response-type> --method <"local"|"global"> <query>
# CLI Arguments
# --data <path-to-data> - Folder containing the .parquet output files from running the Indexer.
# --community_level <community-level> - Community level in the Leiden community hierarchy from which we will load the community reports higher value means we use reports on smaller communities. Default: 2
# --response_type <response-type> - Free form text describing the response type and format, can be anything, e.g. Multiple Paragraphs, Single Paragraph, Single Sentence, List of 3-7 Points, Single Page, Multi-Page Report. Default: Multiple Paragraphs.
# --method <"local"|"global"> - Method to use to answer the query, one of local or global.
#

def ask_question():
    # Define default values
    default_question = "Who is Scrooge, and what are his main relationships?"
    default_method = 'local'
    root_dir = './ragtest2'

    # Prompt the user to input their question or use the default
    question = input(f"Please enter your question (default: '{default_question}'): ").strip()
    if not question:
        question = default_question

    # Prompt the user to choose between local or global method or use the default
    method = input(f"Do you want to query locally or globally? (default: '{default_method}'): ").strip().lower()
    if not method:
        method = default_method

    # Validate method input
    if method not in ['local', 'global']:
        print("Invalid method. Please enter 'local' or 'global'.")
        return

    # Construct the command based on the user's choice
    command = [
        'python', '-m', 'graphrag.query',
        '--root', root_dir,
        '--method', method,
        question
    ]

    # Execute the command
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        print(f"Output:\n{result.stdout}")
    except Exception as e:
        print(f"An error occurred: {e}")
        

if __name__ == '__main__':
    ask_question()
