from fpdf import FPDF
import os

# Create data/books directory if it doesn't exist
os.makedirs("data/books", exist_ok=True)

# Create a sample PDF
pdf = FPDF()

# Add first page
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="This is a test PDF document", ln=True)
pdf.cell(200, 10, txt="Created for testing the PDF RAG system", ln=True)
pdf.cell(200, 10, txt="It contains sample text about blockchain technology", ln=True)
pdf.ln(10)
pdf.cell(200, 10, txt="Blockchain is a distributed ledger technology", ln=True)
pdf.cell(200, 10, txt="that enables secure and transparent transactions", ln=True)
pdf.cell(200, 10, txt="without the need for intermediaries.", ln=True)

# Add second page
pdf.add_page()
pdf.cell(200, 10, txt="Bitcoin is the first and most well-known application", ln=True)
pdf.cell(200, 10, txt="of blockchain technology. It was introduced in 2008", ln=True)
pdf.cell(200, 10, txt="by an anonymous person or group known as Satoshi Nakamoto.", ln=True)
pdf.ln(10)
pdf.cell(200, 10, txt="Ethereum is another popular blockchain platform", ln=True)
pdf.cell(200, 10, txt="that introduced smart contracts, which are self-executing", ln=True)
pdf.cell(200, 10, txt="contracts with the terms directly written into code.", ln=True)

# Save the PDF
output_path = "data/books/blockchain_sample.pdf"
pdf.output(output_path)

print(f"Sample PDF created at: {output_path}") 