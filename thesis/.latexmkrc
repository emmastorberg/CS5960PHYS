# Use pdflatex for PDF generation
$pdf_mode = 1;

# Use biber for bibliography processing (biblatex)
$biber = 'biber %O %S';
$bibtex_use = 2;  # Use biber instead of bibtex

# PDF viewer settings
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';

# Clean up auxiliary files
@generated_exts = (@generated_exts, 'synctex.gz', 'run.xml', 'bbl', 'bcf', 'blg');
