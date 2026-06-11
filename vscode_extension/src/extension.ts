import * as vscode from 'vscode';

const API_BASE = () => vscode.workspace.getConfiguration('archonai').get<string>('apiUrl') || 'http://localhost:8000';
const REPOSITORY = () => vscode.workspace.getConfiguration('archonai').get<string>('repository') || 'langchain';

async function callArchonAI(message: string): Promise<string> {
  const response = await fetch(`${API_BASE()}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, repository: REPOSITORY(), session_id: 'vscode' }),
  });

  if (!response.ok) throw new Error(`API error: ${response.status}`);
  if (!response.body) throw new Error('No response body');

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let result = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    for (const line of buffer.split('\n')) {
      if (line.startsWith('data: ')) {
        try {
          const payload = JSON.parse(line.slice(6));
          if (payload.text) result += payload.text;
        } catch { /* skip */ }
      }
    }
    buffer = buffer.split('\n').pop() || '';
  }
  return result || 'No response received.';
}

function getSelectedCode(): string {
  const editor = vscode.window.activeTextEditor;
  if (!editor) return '';
  return editor.document.getText(editor.selection) || editor.document.getText();
}

export function activate(context: vscode.ExtensionContext) {
  const explain = vscode.commands.registerCommand('archonai.explain', async () => {
    const code = getSelectedCode();
    if (!code) return vscode.window.showWarningMessage('Select code to explain.');
    await vscode.window.withProgress({ location: vscode.ProgressLocation.Notification, title: 'ArchonAI explaining...' }, async () => {
      const result = await callArchonAI(`Explain this code:\n\`\`\`\n${code}\n\`\`\``);
      const doc = await vscode.workspace.openTextDocument({ content: result, language: 'markdown' });
      await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);
    });
  });

  const generate = vscode.commands.registerCommand('archonai.generate', async () => {
    const prompt = await vscode.window.showInputBox({ prompt: 'Describe the code to generate' });
    if (!prompt) return;
    const result = await callArchonAI(`Generate code: ${prompt}`);
    const doc = await vscode.workspace.openTextDocument({ content: result, language: 'python' });
    await vscode.window.showTextDocument(doc);
  });

  const ask = vscode.commands.registerCommand('archonai.ask', async () => {
    const question = await vscode.window.showInputBox({ prompt: 'Ask about the repository' });
    if (!question) return;
    const result = await callArchonAI(question);
    vscode.window.showInformationMessage(result.slice(0, 200) + '...');
  });

  const refactor = vscode.commands.registerCommand('archonai.refactor', async () => {
    const code = getSelectedCode();
    if (!code) return;
    const result = await callArchonAI(`Refactor this code to use modern patterns:\n\`\`\`\n${code}\n\`\`\``);
    const edit = new vscode.WorkspaceEdit();
    const editor = vscode.window.activeTextEditor!;
    edit.replace(editor.document.uri, editor.selection, result);
    await vscode.workspace.applyEdit(edit);
  });

  const fix = vscode.commands.registerCommand('archonai.fix', async () => {
    const code = getSelectedCode();
    const diagnostics = vscode.languages.getDiagnostics(vscode.window.activeTextEditor!.document.uri);
    const errors = diagnostics.map(d => d.message).join('\n');
    const result = await callArchonAI(`Fix errors in this code:\nErrors: ${errors}\n\`\`\`\n${code}\n\`\`\``);
    vscode.window.showInformationMessage('Fix suggestion ready — check output panel.');
    const channel = vscode.window.createOutputChannel('ArchonAI');
    channel.appendLine(result);
    channel.show();
  });

  context.subscriptions.push(explain, generate, ask, refactor, fix);
}

export function deactivate() {}
