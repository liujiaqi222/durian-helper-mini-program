import type { UserConfigExport } from "@tarojs/cli"

export default {
  defineConstants: {
    __API_BASE_URL__: JSON.stringify('http://127.0.0.1:3000/api/v1'),
  },
   logger: {
    quiet: false,
    stats: true
  },
  mini: {},
  h5: {}
} satisfies UserConfigExport<'webpack5'>
