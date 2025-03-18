# Time Series Analysis and Forecasting

Key Topics: Let's look at each one of them, starting with the trend, the general direction of the time series. 
1. **Trend:** Trend can change overtime but not everyday

2. **Seasonality:** These are seasonal cycles so image a time series that is higher during summer and lower during winter. There are two types of seasonality -
1. Addictive: Seasonal cycles are constant in value
2. Multiplicative: Seasonal cycles are **proportional** to the trend

3. **Error:** Error term - Which is what is not explained by the trend and seasonality. It is supposed to be a random walk, without a pattern, like in the chart.

4. Why is understanding whether a time series data exhibits additive or multiplicative seasonality impact? - It assists in making informed decisions and improving the accuracy of future predictions by identifying how seasonal fluctuations interact with the trend.

5. Note:  Best Practices related to time series modeling and forecasting, specifically in setting the seasonality parameter for **different time granularities**.
**Seasonality**
- 24 hours for hourly: Captures daily seasonality (e.g., electricity usage or website traffic peaking at certain hours).
- 7 or 365 for daily, but 7 is preferred.
- 7 is preferred for weekly seasonality (e.g., sales or web traffic patterns that repeat weekly).
- 365 if there’s an annual seasonality component (e.g., temperature patterns or holiday-based fluctuations).
**For modeling**
- 52 for weekly → Captures yearly seasonality since there are 52 weeks in a year.
- 12 for monthly → Accounts for yearly seasonality since there are 12 months in a year.
- 4 for quarterly → Represents one full year divided into 4 quarters.
- 5 for weekdays → Likely used for business-related datasets where only weekdays (Monday–Friday) matter, excluding weekends.
 
