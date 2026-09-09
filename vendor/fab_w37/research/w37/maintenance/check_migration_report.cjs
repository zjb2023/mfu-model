// Reuse established browser checks; write evidence only to the migration audit directory.
const fs=require('fs'),root=require('path').resolve(__dirname,'../../..');
fs.mkdirSync(root+'/results/w37/migration-audit-20260909',{recursive:true});
const original=fs.readFileSync(root+'/research/w37/onef1b/history_scaleout/check_inline.cjs','utf8');
const adapted=original.replace("root=path.resolve(__dirname,'../../../..'),out=root+'/results/w37/A/history-scaleout-20260908'", "root="+JSON.stringify(root)+",out=root+'/results/w37/migration-audit-20260909'").replace("'file://'+out+'/integrated_report.html#scaleout'","'file://'+root+'/results/w37/A/history-scaleout-20260908/integrated_report.html#scaleout'");
if(adapted===original)throw Error('Upstream check layout changed; review adapter');
eval(adapted);
