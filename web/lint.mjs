// Static checks the browser test cannot catch, run before every build.
import fs from 'fs';
const src = fs.readFileSync('app.template.html', 'utf8');
const problems = [];

const nonAscii = [...src].filter(c => c.codePointAt(0) > 127);
if (nonAscii.length) {
  const set = [...new Set(nonAscii)].join(' ');
  problems.push(`${nonAscii.length} non-ASCII characters (${set}). They render as mojibake wherever no charset is declared. Use HTML entities, or a \\uXXXX escape inside a JS string assigned through textContent.`);
}
// A shell heredoc quoting a JS string can leave \\' , which ends the string early
// and takes the whole <script> block down with it.
if (/\\\\'/.test(src)) problems.push('found \\\\\' in source: a double-escaped quote will terminate a JS string early');
for (const token of ['__DATA__', '__LOGOS__']) {
  if (!src.includes(token)) problems.push(`template is missing the ${token} placeholder`);
}
// The boot call has to be the last thing that runs or late page modules are undefined.
const boot = src.lastIndexOf('window.PYLON_BOOT()');
const lastRegister = src.lastIndexOf('registerPage(');
if (boot < lastRegister) problems.push('window.PYLON_BOOT() runs before the last registerPage() call');

if (problems.length) { console.error('lint failed:\n  - ' + problems.join('\n  - ')); process.exit(1); }
console.log('lint passed: ASCII-clean, placeholders present, boot ordered last');
