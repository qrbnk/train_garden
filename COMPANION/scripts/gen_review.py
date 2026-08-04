import sys
import os
import argparse

def generate_viewer(workspace, skill_name, output_path):
    # Find the markdown file the AI just wrote
    outputs_dir = os.path.join(workspace, "run_001", "outputs")
    md_files = [f for f in os.listdir(outputs_dir) if f.endswith('.md')]
    
    if not md_files:
        print("No markdown files found to convert!")
        return

    with open(os.path.join(outputs_dir, md_files[0]), 'r') as f:
        content = f.read()

    # Create a simple, clean HTML layout
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Eval Review: {skill_name}</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-slate-900 text-white p-10">
        <div class="max-w-4xl mx-auto">
            <h1 class="text-3xl font-bold mb-6 border-b border-slate-700 pb-4">Skill Benchmark: {skill_name}</h1>
            <div class="bg-slate-800 p-8 rounded-2xl shadow-xl prose prose-invert lg:prose-xl">
                {content.replace('#', '##').replace('\\n', '<br>')}
            </div>
            <div class="mt-10 text-slate-500 text-sm italic">
                Generated via Train Garden AI Protocol 2026
            </div>
        </div>
    </body>
    </html>
    """

    with open(output_path, 'w') as f:
        f.write(html_content)
    print(f"✅ Viewer successfully built at: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace")
    parser.add_argument("--skill-name")
    parser.add_argument("--static")
    args = parser.parse_args()
    
    generate_viewer(args.workspace, args.skill_name, args.static)