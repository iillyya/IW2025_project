import os
import argparse
from pathlib import Path
from google import genai
import PyPDF2

def extract_text_from_pdf(pdf_path):
  """Extract text content from a PDF file"""
  pdf_path = Path(pdf_path)
  if not pdf_path.exists():
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

  try:
    with open(pdf_path, 'rb') as file:
      pdf_reader = PyPDF2.PdfReader(file)
      text = "\n".join(page.extract_text() for page in pdf_reader.pages)
      return text
  except Exception as e:
    raise Exception(f"Error extracting text from PDF: {e}")

def ask_gemini(prompt, text_content, api_key=None):
  """Send query to Gemini API and return response"""
  # Use provided API key or get from environment
  api_key = api_key or os.environ.get('GEMINI_API_KEY')
  if not api_key:
    raise ValueError("Gemini API key not provided")

  try:
    client = genai.Client(api_key=api_key)
    full_prompt = f"{prompt}\n\nHere is the document content:\n{text_content}"

    response = client.models.generate_content(
      model="gemini-2.5-flash",
      contents=full_prompt
    )

    return response.text
  except Exception as e:
    raise Exception(f"Error calling Gemini API: {e}")

def get_analysis_prompt():
  """Return the prompt template for material analysis"""
  return """
  Представь, что ты специалист по материаловедению. Проанализируй pdf-документ, который представляет из себя статью по некоторому сплаву и извлеки следующую информацию об этом сплаве:

  1. Механические свойства
  Прочность на разрыв (Tensile strength) – максимальная нагрузка перед разрушением.
  Предел текучести (Yield strength) – напряжение, при котором материал начинает пластически деформироваться.
  Твёрдость (Hardness) – сопротивление материала локальной деформации (например, по Бринеллю, Роквеллу).
  Ударная вязкость (Impact toughness) – способность поглощать энергию при ударе.
  Эластичность / модуль Юнга (Elastic modulus) – жёсткость материала.

  2. Физические свойства
  Плотность (Density) – масса на единицу объёма.
  Теплопроводность (Thermal conductivity) – важна для теплообмена.
  Электропроводность (Electrical conductivity / resistivity) – для сплавов, используемых в электронике.
  Температура плавления / термическая стабильность – диапазон рабочих температур.

  3. Химические свойства
  Коррозионная стойкость – скорость коррозии в различных средах.
  Химическая устойчивость – реакция с кислотами, щелочами, газами.
  Состав (Chemical composition) – процентное содержание основных и легирующих элементов.

  4. Металлургические свойства
  Структура зерна (Grain structure / size) – влияет на механические свойства.
  Фазы и кристаллическая структура – наличие α, β, γ фаз и т.д.
  Твердость/прочность после термообработки – зависимость свойств от закалки, отпусков.
  Скорость старения / релаксации напряжений – особенно для алюминиевых и титансодержащих сплавов.

  5. Дополнительные эксплуатационные характеристики
  Износостойкость – сопротивление трению и истиранию.
  Усталостная прочность (Fatigue strength) – способность выдерживать циклические нагрузки.
  Ковкость и пластичность – возможность деформироваться без разрушения.
  Магнитные свойства – для специальных сплавов (например, ферромагнитные).

  ВАЖНО: если определенная информация в статье отсутствует, то напиши ИНФОРМАЦИЯ ОТСУТСТВУЕТ.
  """

def main():
  parser = argparse.ArgumentParser(description="Extract and analyze information from PDF using Gemini AI")
  parser.add_argument("--pdf-path", help="Path to the PDF file")
  parser.add_argument("--output", "-o", default=None, help="Output file path (default: extracted_<filename>.txt)")
  parser.add_argument("--api-key", help="Gemini API key (can also use GEMINI_API_KEY env variable)")
  args = parser.parse_args()

  # Set API key if provided via arguments
  if args.api_key:
    os.environ['GEMINI_API_KEY'] = args.api_key

  try:
    # Extract text from PDF
    print(f"Extracting text from {args.pdf_path}...")
    pdf_text = extract_text_from_pdf(args.pdf_path)
    print(f"Extracted {len(pdf_text)} characters")

    # Get the analysis prompt
    prompt = get_analysis_prompt()

    # Generate default output filename if not specified
    if not args.output:
      pdf_name = Path(args.pdf_path).stem
      output_file = f"extracted_{pdf_name}.txt"
    else:
      output_file = args.output

    # Get response from Gemini
    print("Analyzing content with Gemini AI...")
    response = ask_gemini(prompt, pdf_text)

    # Save and display results
    print("\nExtracted Information:")
    print("=" * 50)
    print(response)

    with open(output_file, "w", encoding="utf-8") as f:
      f.write(response)
    print(f"\nResults saved to '{output_file}'")

  except Exception as e:
    print(f"Error: {e}")
    return 1

  return 0

if __name__ == "__main__":
  main()
