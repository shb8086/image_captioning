#!/bin/bash

# Compile the LaTeX file to generate PDF
pdflatex main.tex

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "PDF generated successfully."

    # Remove temporary files
    rm -f main.aux main.log main.fls main.fdb_latexmk main.synctex.gz

    echo "Temporary files removed."
else
    echo "Error: LaTeX compilation failed."
fi
