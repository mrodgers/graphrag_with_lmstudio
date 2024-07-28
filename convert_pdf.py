from langchain_community.document_loaders import PyPDFLoader


def pdf_to_text(pdf_path, txt_path):
    # Initialize the PDF loader
    loader = PyPDFLoader(file_path=pdf_path)

    # Load the PDF into a list of documents
    pages = loader.load_and_split()

    # Open a text file in write mode
    with open(txt_path, 'w', encoding='utf-8') as txt_file:
        # Iterate through each document (page)
        for page in pages:
            text = page.page_content  # Extract text from page content
            txt_file.write(text)  # Write text to file
            txt_file.write("\n\n")  # Add a newline between pages

    print(f'Text extracted from {pdf_path} and saved to {txt_path}')

# Example usage
pdf_path = 'file.pdf'  # Replace with your PDF file path
txt_path = './cdfmc/input/cdfmc_docs.txt'  # Replace with your desired text file path
pdf_to_text(pdf_path, txt_path)
