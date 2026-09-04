import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

const suiteDir = path.dirname(fileURLToPath(import.meta.url)).replace(/\\/g, '/');
const openWebui = resolveOpenWebui();

// OPEN_WEBUI_SOURCE_DIR points at the backend, as for the python suites; otherwise a sibling checkout.
function resolveOpenWebui(): string {
	const backend = process.env.OPEN_WEBUI_SOURCE_DIR;
	if (backend) return path.resolve(backend, '..');
	for (let dir = suiteDir; path.dirname(dir) !== dir; dir = path.dirname(dir)) {
		const candidate = path.join(dir, 'open-webui');
		if (fs.existsSync(path.join(candidate, 'package.json'))) return candidate;
	}
	throw new Error('open-webui checkout not found, set OPEN_WEBUI_SOURCE_DIR');
}

const isBare = (id: string) =>
	!id.startsWith('.') && !id.startsWith('\0') && !id.startsWith('node:') && !path.isAbsolute(id);

export default defineConfig({
	root: suiteDir,
	resolve: { alias: { $lib: path.join(openWebui, 'src', 'lib') } },
	plugins: [
		{
			// Bare imports in a test resolve from the checkout's node_modules, so test and module under test share one i18next.
			name: 'resolve-from-open-webui',
			enforce: 'pre',
			resolveId(id, importer) {
				const fromSuite = importer?.startsWith(suiteDir) && !importer.includes('/node_modules/');
				if (!fromSuite || !isBare(id) || id === 'vitest' || id.startsWith('@vitest/')) return null;
				return this.resolve(id, path.join(openWebui, 'package.json'), { skipSelf: true });
			}
		}
	]
});
