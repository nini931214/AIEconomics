193
194
195
196
197
198
199
200
201
202
203
204
205
206
207
208
209
210
211
212
213
214
215
216
217
218
219
220
221
222
223
224
225
226
227
228
229
230
231
232
233
234
235
236
237
238
239
240
241
242
243
244
245
246
247
248
249
250
251
252
253
254
255
256
257
258
259
260
261
262
263
264
265
266
267
268
269
270
271
272
273
274
275
276
277
278
279
280
281
282
283
284
285
286
287
288
289
290
291
292
293
294
295
else:
    st.info('取得 DGS10 與 DGS2 後會顯示倒掛燈號與歷史圖。')

st.subheader('📈 金融市場與總經歷史趨勢')
tab_market, tab_macro = st.tabs(['金融市場（Yahoo Finance）', '總體經濟（FRED）'])
with tab_market:
    if not market_df.empty:
        choice = st.selectbox('市場標的', list(market_df.columns))
        fig = px.line(market_df, y=choice, title=f'{choice}｜最近兩年可用行情')
        fig.update_layout(xaxis_title='日期', yaxis_title='行情數值')
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f'最新觀測日：{market_df[choice].dropna().index[-1].date()}；Yahoo Finance 行情可能延遲。')
    else:
        st.warning('目前無法取得市場行情。')
with tab_macro:
    if not macro_df.empty:
        choice2 = st.selectbox('總體指標', list(macro_df.columns))
        fig = px.line(macro_df[choice2].dropna(), title=f'{choice2}｜歷史觀測值')
        fig.update_layout(xaxis_title='日期', yaxis_title='原始單位')
        st.plotly_chart(fig, use_container_width=True)
        st.caption('各指標有不同發布頻率；未對缺值做跨頻率填補。')
    else:
        st.info('請提供 FRED API Key。')

st.subheader('💱 美元、台幣與英鎊的連動')
USD, GBPUSD, GBPTWD = 'USD/TWD（1 美元＝台幣）', 'GBP/USD（1 英鎊＝美元）', 'GBP/TWD（1 英鎊＝台幣）'
if not market_df.empty:
    cols = st.columns(3)
    metric(cols[0], '1 美元＝新台幣', market_df, USD, ' TWD', 3)
    metric(cols[1], '1 英鎊＝美元', market_df, GBPUSD, ' USD', 4)
    metric(cols[2], '1 英鎊＝新台幣', market_df, GBPTWD, ' TWD', 3)
    if all(x in market_df for x in (USD, GBPUSD, GBPTWD)):
        pair = market_df[[USD, GBPUSD, GBPTWD]].ffill(limit=3).dropna()
        if len(pair) >= 20:
            normalized = pair / pair.iloc[0] * 100
            fig = px.line(normalized, title='三組匯率走勢比較（共同起點＝100）')
            fig.update_layout(yaxis_title='指數（起點＝100）', xaxis_title='日期')
            st.plotly_chart(fig, use_container_width=True)
            daily_changes = pair.pct_change(fill_method=None).dropna() * 100
            if len(daily_changes) >= 20:
                corr = daily_changes.corr()
                st.plotly_chart(px.imshow(corr, text_auto='.2f', zmin=-1, zmax=1, color_continuous_scale='RdBu_r', title='每日匯率變動相關係數'), use_container_width=True)
            st.markdown('**匯率恆等式：GBP/TWD ≈ GBP/USD × USD/TWD。** 美元對台幣升值可能推高英鎊換匯成本，但若英鎊同時對美元貶值，兩者可能抵銷。相關性不是因果關係。')
            st.caption('這是市場參考行情，不是台新銀行即期賣出價；實際購匯成本需另查銀行牌告。')
            sc = scenario(pair[GBPTWD])
            if sc:
                st.markdown(f"**30 個交易日歷史波動情境（非價格預測）：** 假設價格不變且日報酬波動延續，約為 **{sc['low']:.2f}–{sc['high']:.2f} TWD/GBP**；基準報價 {sc['spot']:.2f}。")
                st.caption('採最近 90 筆日對數報酬標準差 × √30 與 ±1.96 倍波動幅度；未納入央行決策、突發新聞、趨勢漂移及極端風險，不能當作 95% 實際預測保證。')

st.subheader('📰 每日財經時事與央行動向')
news = get_news()
if news:
    for item in news:
        st.markdown(f"- [{item['title']}]({item['url']})  ·  {item['published']}")
    st.caption('Google News RSS 聚合標題；新聞日期與來源請點開核實。新聞不代表已確認的市場因果解釋。')
else:
    st.warning('目前無法讀取新聞 RSS；請稍後刷新。')

st.subheader('🤖 Gemini AI 總體經濟摘要')
st.caption('按下按鈕後才會呼叫 Gemini。摘要包含最新觀測值與日期、利差警示、匯率與新聞標題；不保證市場預測準確。')
if st.button('✨ 產生 AI 總體經濟摘要', type='primary'):
    if not gemini_key:
        st.warning('請先在側邊欄或 .streamlit/secrets.toml 設定 GEMINI_API_KEY。')
    elif macro_df.empty:
        st.warning('需要 FRED API Key 與可用總經資料，才能產生摘要。')
    else:
        context = []
        for key in ['美國 CPI 指數', '美國失業率', '美國 10 年期公債殖利率', '美國 2 年期公債殖利率', '聯邦基金有效利率', '日本 10 年期公債殖利率', spread_key]:
            if key in macro_df:
                value, dt, _ = last_obs(macro_df[key])
                if value is not None:
                    context.append(f'{key}: {value:.3f}（觀測日 {dt.date()}）')
        for key in [USD, GBPUSD, GBPTWD]:
            if key in market_df:
                value, dt, _ = last_obs(market_df[key])
                if value is not None:
                    context.append(f'{key}: {value:.4f}（行情日 {dt.date()}）')
        context.extend('新聞標題（未核實全文）：' + x['title'] + ' / ' + x['published'] for x in news[:5])
        prompt = ('請以繁體中文撰寫約 200 個中文字的總體經濟短評，重點是美國通膨、就業、利率、10Y−2Y 利差，'
                  '以及美元/台幣與英鎊/台幣的連動，兼顧英國留學換匯者。務必區分各指標觀測日與新聞日期；'
                  'CPI 指數不是通膨率，日本殖利率為月頻，不能稱為即時數值；'
                  '新聞標題未驗證全文，不能把標題直接當因果證據；不得杜撰政策、數字、即時事件或確定的匯率預測。'
                  '如果資料不足，直接指出。\n\n資料：\n' + '\n'.join(context))
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            with st.spinner('Gemini 正在撰寫摘要…'):
                response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
            if response.text:
                st.markdown(response.text)
            else:
                st.warning('Gemini 沒有回傳文字，請稍後再試。')
        except Exception as exc:
            st.error(f'Gemini 呼叫失敗：{exc}')

with st.expander('資料來源、缺漏與更新頻率'):
    st.markdown('FRED：美國 CPI／失業率／有效聯邦基金利率通常為月資料；DGS10、DGS2 通常為日資料；日本 10 年公債系列 IRLTLT01JPM156N 為月資料。Yahoo Finance 為市場參考行情，可能延遲或缺漏。新聞為 Google News RSS。')
    if macro_errors:
        st.warning('FRED 部分系列失敗：' + '；'.join(macro_errors.keys()))
    if market_errors:
        st.warning('市場資料備註：' + '；'.join(f'{k}: {v}' for k, v in market_errors.items()))
    st.markdown('[FRED](https://fred.stlouisfed.org/) · [Yahoo Finance](https://finance.yahoo.com/) · [台新銀行](https://www.taishinbank.com.tw/)')
