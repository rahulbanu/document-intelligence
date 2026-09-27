from services.annotation_service import detect_pdf_annotations


pdf_path = "uploads/annotation_test_document.pdf"

result = detect_pdf_annotations(pdf_path)

print("\nANNOTATION DETECTION RESULT\n")

for page in result["pages"]:

    print(f"Page {page['page_number']}")

    if not page["annotations"]:
        print("  No annotations found.")

    for annotation in page["annotations"]:

        print("  Type:", annotation["type"])
        print("  Text:", annotation["text"])