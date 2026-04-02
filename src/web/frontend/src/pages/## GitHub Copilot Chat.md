## GitHub Copilot Chat

- Extension: 0.37.9 (prod)
- VS Code: 1.109.5 (072586267e68ece9a47aa43f8c108e0dcbf44622)
- OS: darwin 25.2.0 arm64
- GitHub Account: sebastian-siemianowski

## Network

User Settings:
```json
  "http.systemCertificatesNode": false,
  "github.copilot.advanced.debug.useElectronFetcher": true,
  "github.copilot.advanced.debug.useNodeFetcher": false,
  "github.copilot.advanced.debug.useNodeFetchFetcher": true
```

Connecting to https://api.github.com:
- DNS ipv4 Lookup: 140.82.121.5 (1 ms)
- DNS ipv6 Lookup: ::ffff:140.82.121.5 (2 ms)
- Proxy URL: None (0 ms)
- Electron fetch (configured): Error (2 ms): Error: net::ERR_ADDRESS_INVALID
	at SimpleURLLoaderWrapper.<anonymous> (node:electron/js2c/utility_init:2:10684)
	at SimpleURLLoaderWrapper.emit (node:events:519:28)
	at SimpleURLLoaderWrapper.emit (node:domain:489:12)
  [object Object]
  {"is_request_error":true,"network_process_crashed":false}
- Node.js https: Error (10 ms): Error: connect EADDRNOTAVAIL 140.82.121.5:443 - Local (0.0.0.0:0)
	at internalConnect (node:net:1110:16)
	at defaultTriggerAsyncIdScope (node:internal/async_hooks:472:18)
	at GetAddrInfoReqWrap.emitLookup [as callback] (node:net:1523:9)
	at GetAddrInfoReqWrap.onlookupall [as oncomplete] (node:dns:134:8)
- Node.js fetch: Error (18 ms): TypeError: fetch failed
	at node:internal/deps/undici/undici:14900:13
	at process.processTicksAndRejections (node:internal/process/task_queues:105:5)
	at async n._fetch (/Users/sebastiansiemianowski/.vscode/extensions/github.copilot-chat-0.37.9/dist/extension.js:4862:26129)
	at async n.fetch (/Users/sebastiansiemianowski/.vscode/extensions/github.copilot-chat-0.37.9/dist/extension.js:4862:25777)
	at async u (/Users/sebastiansiemianowski/.vscode/extensions/github.copilot-chat-0.37.9/dist/extension.js:4894:190)
	at async CA.h (file:///Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/workbench/api/node/extensionHostProcess.js:116:41743)
  Error: connect EADDRNOTAVAIL 140.82.121.5:443 - Local (0.0.0.0:0)
  	at internalConnect (node:net:1110:16)
  	at defaultTriggerAsyncIdScope (node:internal/async_hooks:472:18)
  	at GetAddrInfoReqWrap.emitLookup [as callback] (node:net:1523:9)
  	at GetAddrInfoReqWrap.onlookupall [as oncomplete] (node:dns:134:8)

Connecting to https://api.githubcopilot.com/_ping:
- DNS ipv4 Lookup: 140.82.113.22 (37 ms)
- DNS ipv6 Lookup: ::ffff:140.82.113.22 (2 ms)
- Proxy URL: None (1 ms)
- Electron fetch (configured): Error (15 ms): Error: net::ERR_ADDRESS_INVALID
	at SimpleURLLoaderWrapper.<anonymous> (node:electron/js2c/utility_init:2:10684)
	at SimpleURLLoaderWrapper.emit (node:events:519:28)
	at SimpleURLLoaderWrapper.emit (node:domain:489:12)
  [object Object]
  {"is_request_error":true,"network_process_crashed":false}
- Node.js https: Error (6 ms): Error: connect EADDRNOTAVAIL 140.82.113.22:443 - Local (0.0.0.0:0)
	at internalConnect (node:net:1110:16)
	at defaultTriggerAsyncIdScope (node:internal/async_hooks:472:18)
	at GetAddrInfoReqWrap.emitLookup [as callback] (node:net:1523:9)
	at GetAddrInfoReqWrap.onlookupall [as oncomplete] (node:dns:134:8)
- Node.js fetch: Error (7 ms): TypeError: fetch failed
	at node:internal/deps/undici/undici:14900:13
	at process.processTicksAndRejections (node:internal/process/task_queues:105:5)
	at async n._fetch (/Users/sebastiansiemianowski/.vscode/extensions/github.copilot-chat-0.37.9/dist/extension.js:4862:26129)
	at async n.fetch (/Users/sebastiansiemianowski/.vscode/extensions/github.copilot-chat-0.37.9/dist/extension.js:4862:25777)
	at async u (/Users/sebastiansiemianowski/.vscode/extensions/github.copilot-chat-0.37.9/dist/extension.js:4894:190)
	at async CA.h (file:///Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/workbench/api/node/extensionHostProcess.js:116:41743)
  Error: connect EADDRNOTAVAIL 140.82.113.22:443 - Local (0.0.0.0:0)
  	at internalConnect (node:net:1110:16)
  	at defaultTriggerAsyncIdScope (node:internal/async_hooks:472:18)
  	at GetAddrInfoReqWrap.emitLookup [as callback] (node:net:1523:9)
  	at GetAddrInfoReqWrap.onlookupall [as oncomplete] (node:dns:134:8)

Connecting to https://copilot-proxy.githubusercontent.com/_ping:
- DNS ipv4 Lookup: 20.250.119.64 (52 ms)
- DNS ipv6 Lookup: ::ffff:20.250.119.64 (2 ms)
- Proxy URL: None (2 ms)
- Electron fetch (configured): Error (14 ms): Error: net::ERR_ADDRESS_INVALID
	at SimpleURLLoaderWrapper.<anonymous> (node:electron/js2c/utility_init:2:10684)
	at SimpleURLLoaderWrapper.emit (node:events:519:28)
	at SimpleURLLoaderWrapper.emit (node:domain:489:12)
  [object Object]
  {"is_request_error":true,"network_process_crashed":false}
- Node.js https: Error (7 ms): Error: connect EADDRNOTAVAIL 20.250.119.64:443 - Local (0.0.0.0:0)
	at internalConnect (node:net:1110:16)
	at defaultTriggerAsyncIdScope (node:internal/async_hooks:472:18)
	at GetAddrInfoReqWrap.emitLookup [as callback] (node:net:1523:9)
	at GetAddrInfoReqWrap.onlookupall [as oncomplete] (node:dns:134:8)
- Node.js fetch: Error (9 ms): TypeError: fetch failed
	at node:internal/deps/undici/undici:14900:13
	at process.processTicksAndRejections (node:internal/process/task_queues:105:5)
	at async n._fetch (/Users/sebastiansiemianowski/.vscode/extensions/github.copilot-chat-0.37.9/dist/extension.js:4862:26129)
	at async n.fetch (/Users/sebastiansiemianowski/.vscode/extensions/github.copilot-chat-0.37.9/dist/extension.js:4862:25777)
	at async u (/Users/sebastiansiemianowski/.vscode/extensions/github.copilot-chat-0.37.9/dist/extension.js:4894:190)
	at async CA.h (file:///Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/workbench/api/node/extensionHostProcess.js:116:41743)
  Error: connect EADDRNOTAVAIL 20.250.119.64:443 - Local (0.0.0.0:0)
  	at internalConnect (node:net:1110:16)
  	at defaultTriggerAsyncIdScope (node:internal/async_hooks:472:18)
  	at GetAddrInfoReqWrap.emitLookup [as callback] (node:net:1523:9)
  	at GetAddrInfoReqWrap.onlookupall [as oncomplete] (node:dns:134:8)

Connecting to https://mobile.events.data.microsoft.com: Error (2 ms): Error: net::ERR_ADDRESS_INVALID
	at SimpleURLLoaderWrapper.<anonymous> (node:electron/js2c/utility_init:2:10684)
	at SimpleURLLoaderWrapper.emit (node:events:519:28)
	at SimpleURLLoaderWrapper.emit (node:domain:489:12)
  [object Object]
  {"is_request_error":true,"network_process_crashed":false}
Connecting to https://dc.services.visualstudio.com: Error (51 ms): Error: net::ERR_ADDRESS_INVALID
	at SimpleURLLoaderWrapper.<anonymous> (node:electron/js2c/utility_init:2:10684)
	at SimpleURLLoaderWrapper.emit (node:events:519:28)
	at SimpleURLLoaderWrapper.emit (node:domain:489:12)
  [object Object]
  {"is_request_error":true,"network_process_crashed":false}
Connecting to https://copilot-telemetry.githubusercontent.com/_ping: Error (7 ms): Error: connect EADDRNOTAVAIL 140.82.113.22:443 - Local (0.0.0.0:0)
	at internalConnect (node:net:1110:16)
	at defaultTriggerAsyncIdScope (node:internal/async_hooks:472:18)
	at GetAddrInfoReqWrap.emitLookup [as callback] (node:net:1523:9)
	at GetAddrInfoReqWrap.onlookupall [as oncomplete] (node:dns:134:8)
Connecting to https://copilot-telemetry.githubusercontent.com/_ping: Error (6 ms): Error: connect EADDRNOTAVAIL 140.82.113.22:443 - Local (0.0.0.0:0)
	at internalConnect (node:net:1110:16)
	at defaultTriggerAsyncIdScope (node:internal/async_hooks:472:18)
	at GetAddrInfoReqWrap.emitLookup [as callback] (node:net:1523:9)
	at GetAddrInfoReqWrap.onlookupall [as oncomplete] (node:dns:134:8)
Connecting to https://default.exp-tas.com: Error (41 ms): Error: connect EADDRNOTAVAIL 13.107.5.93:443 - Local (0.0.0.0:0)
	at internalConnect (node:net:1110:16)
	at defaultTriggerAsyncIdScope (node:internal/async_hooks:472:18)
	at GetAddrInfoReqWrap.emitLookup [as callback] (node:net:1523:9)
	at GetAddrInfoReqWrap.onlookupall [as oncomplete] (node:dns:134:8)

Number of system certificates: 10

## Documentation

In corporate networks: [Troubleshooting firewall settings for GitHub Copilot](https://docs.github.com/en/copilot/troubleshooting-github-copilot/troubleshooting-firewall-settings-for-github-copilot).