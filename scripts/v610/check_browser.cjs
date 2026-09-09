// Reuse the frozen UI assertions, changing only environment and output paths.
const fs = require('fs'), path = require('path');
const root = path.resolve(__dirname, '../..');
const service = JSON.parse(fs.readFileSync(path.join(root, 'results/v610/service.json')));
const origin = `http://127.0.0.1:${service.port}`;
const kind = process.argv[2] || 'release';
const out = path.join(root, 'results/v610/webcheck-' + kind);
fs.mkdirSync(out, {recursive: true});
function replaceRequired(text, from, to) {
  if (!text.includes(from)) throw Error('Frozen browser adapter anchor changed: ' + from);
  return text.replaceAll(from, to);
}
let text;
if (kind === 'release') {
  text = fs.readFileSync(path.join(root, 'vendor/fab_w37/research/w37/onef1b/v610_browser.cjs'), 'utf8');
  text = replaceRequired(text, "root=path.resolve(__dirname,'../../..'),out=path.join(root,'results/w37/A/v610-webcheck')", `root=${JSON.stringify(root)},out=${JSON.stringify(out)}`);
  text = replaceRequired(text, "require(path.join(root,'results/w37/A/research-html-20260907/browser-check/node_modules/playwright'))", "require('playwright')");
  text = replaceRequired(text, "path.join(root,'results/w37/A/v610-release/render/index.html')", JSON.stringify(path.join(service.run_root, 'render/index.html')));
} else if (kind === 'history') {
  text = fs.readFileSync(path.join(root, 'vendor/fab_w37/research/w37/onef1b/history_scaleout/check_inline.cjs'), 'utf8');
  text = replaceRequired(text, "root=path.resolve(__dirname,'../../../..'),out=root+'/results/w37/A/history-scaleout-20260908'", `root=${JSON.stringify(root)},out=${JSON.stringify(out)}`);
  text = replaceRequired(text, "require(root+'/results/w37/A/research-html-20260907/browser-check/node_modules/playwright')", "require('playwright')");
  const m = JSON.parse(fs.readFileSync(path.join(root, 'docs/v610/artifact_manifest.json')));
  const base = process.env.MFU_V610_ARTIFACT_ROOT || JSON.parse(fs.readFileSync(path.join(root, '.local/v610.json'))).artifact_root;
  const logical = '/home/zjb/Desktop/worktrees/fab-w37-1f1b/results/w37/A/history-scaleout-20260908/integrated_report.html';
  text = replaceRequired(text, "'file://'+out+'/integrated_report.html#scaleout'", JSON.stringify('file://' + path.join(base, m.logical_paths[logical]) + '#scaleout'));
} else { throw Error('Choose release or history'); }
text = replaceRequired(text, 'http://192.168.0.49:8037', origin);
fs.writeFileSync(path.join(out, 'adapted_check.cjs'), text);
new Function('require', '__dirname', text)(require, __dirname);
