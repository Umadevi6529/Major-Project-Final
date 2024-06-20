import pandas as pd

def calculate_cell_match_rate(pdf_table_data, excel_table_data):
    # Count the number of cells in the original PDF table
    total_pdf_cells = pdf_table_data.size
    
    # Count the number of matching cells between the PDF and Excel tables
    matching_cells = sum(pdf_table_data.values.flatten() == excel_table_data.values.flatten())
    
    # Calculate the cell match rate
    cell_match_rate = (matching_cells / total_pdf_cells) * 100
    
    return cell_match_rate

# Example usage
pdf_table_data = pd.read_excel('Result correct.xlsx')  # Read original PDF table data
excel_table_data = pd.read_excel('Result.xlsx')  # Read converted Excel table data

cell_match_rate = calculate_cell_match_rate(pdf_table_data, excel_table_data)
print("Cell Match Rate:", cell_match_rate, "%")
